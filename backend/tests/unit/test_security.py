from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import (
    LoginThrottle,
    TokenSigner,
    check_hash_format,
    hash_password,
    verify_password,
)

PW = "correct horse battery"


@pytest.fixture(scope="module")
def pw_hash() -> str:
    return hash_password(PW)


def test_password_hash_roundtrip(pw_hash: str) -> None:
    assert pw_hash.startswith("scrypt:") and "$" not in pw_hash
    check_hash_format(pw_hash)
    assert verify_password(PW, pw_hash)
    assert not verify_password(PW + " ", pw_hash)
    assert hash_password(PW) != pw_hash  # random salt
    assert not verify_password(PW, "garbage")
    assert not verify_password(PW, "bcrypt:1:2:3:aa:bb")
    with pytest.raises(ValueError):
        hash_password("short")
    with pytest.raises(ValueError):
        check_hash_format("plaintext-password")


def test_token_lifecycle(pw_hash: str) -> None:
    signer = TokenSigner(b"k" * 32, pw_hash, ttl_seconds=3600)
    token, exp = signer.issue(now=1_000_000)
    assert exp == 1_003_600
    assert signer.verify(token, now=1_000_001)
    assert not signer.verify(token, now=1_003_600)  # expired
    body, sig = token.split(".")
    assert not signer.verify(f"{body}.{sig[:-2]}AA", now=1_000_001)  # bad signature
    assert not signer.verify(body, now=1_000_001)
    assert not signer.verify("", now=1_000_001)
    # A different key, or a changed password, invalidates existing tokens.
    assert not TokenSigner(b"x" * 32, pw_hash, 3600).verify(token, now=1_000_001)
    assert not TokenSigner(b"k" * 32, hash_password(PW), 3600).verify(token, now=1_000_001)


def test_login_throttle() -> None:
    t = LoginThrottle(max_failures=3, lockout_seconds=60)
    for i in range(3):
        assert t.retry_after("a", now=100 + i) == 0
        t.failed("a", now=100 + i)
    assert 55 <= t.retry_after("a", now=103) <= 60
    assert t.retry_after("b", now=103) == 0  # per client
    assert t.retry_after("a", now=161) == 0  # oldest failure aged out
    t.failed("c", now=0)
    t.succeeded("c")
    assert t.retry_after("c", now=1) == 0


def test_prod_requires_auth(pw_hash: str) -> None:
    with pytest.raises(ValidationError, match="SWARM_AUTH_PASSWORD_HASH"):
        Settings(_env_file=None, environment="prod")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="SWARM_SECRET_KEY"):
        Settings(_env_file=None, environment="prod", auth_password_hash=pw_hash)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="example values"):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            environment="prod",
            auth_password_hash="scrypt:16384:8:1:REPLACE:REPLACE",
            secret_key="replace-with-the-output-of-swarm-gen-secret-0123456789",
        )
    with pytest.raises(ValidationError, match="32 characters"):
        Settings(_env_file=None, secret_key="short")  # type: ignore[call-arg]
    ok = Settings(  # type: ignore[call-arg]
        _env_file=None, environment="prod", auth_password_hash=pw_hash, secret_key="s" * 40
    )
    assert ok.auth_enabled
    opted_out = Settings(_env_file=None, environment="prod", auth_disabled=True)  # type: ignore[call-arg]
    assert not opted_out.auth_enabled
    assert not Settings(_env_file=None).auth_enabled  # type: ignore[call-arg]
