"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.models.backtest import BacktestResult, Trade
from app.models.champion import MAX_ACTIVE_CHAMPIONS, Champion
from app.models.context_rule import ContextRule
from app.models.dataset import Dataset, Job
from app.models.evolution import EvolutionRun, Generation
from app.models.instrument import Instrument
from app.models.signal import Signal
from app.models.strategy import Strategy, genome_hash, strategy_codename, strategy_lineage

__all__ = [
    "MAX_ACTIVE_CHAMPIONS",
    "BacktestResult",
    "Champion",
    "ContextRule",
    "Dataset",
    "EvolutionRun",
    "Generation",
    "Instrument",
    "Job",
    "Signal",
    "Strategy",
    "Trade",
    "genome_hash",
    "strategy_codename",
    "strategy_lineage",
]
