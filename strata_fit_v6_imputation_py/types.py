from enum import Enum
from .imputation_strategies.mean import MeanImputer

class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"

class NoiseType(str, Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    POISSON = "POISSON"

class ImputationStrategyEnum(str, Enum):
    MEAN = "mean"
    MODE = "mode"
    # TODO
    # other strategies

# Hyperparameters for column names.
# After preprocessing, the survival data will always have these standardized column names.
MINIMUM_ORGANIZATIONS = 3

STRATEGY_MAP = {  
    ImputationStrategyEnum.MEAN.value: MeanImputer,
}