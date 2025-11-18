from enum import Enum

class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"

class NoiseType(str, Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    POISSON = "POISSON"

# Hyperparameters for column names.
# After preprocessing, the survival data will always have these standardized column names.
MINIMUM_ORGANIZATIONS = 3