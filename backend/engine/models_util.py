"""Small helpers shared by the engine and the app layer."""

from __future__ import annotations

from engine.strategy.genome import Genome


def genome_key(genome: Genome) -> str:
    """Stable identity of a strategy: the same hash as ``strategies.genome_hash``."""
    from app.models.strategy import genome_hash

    return genome_hash(genome.canonical())
