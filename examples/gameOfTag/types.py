from enum import Enum


class AgentType(str, Enum):
    PREDATOR = "predator"
    PREY = "prey"


class Mode(str, Enum):
    EVALUATE = "evaluate"
    TRAIN = "train"
