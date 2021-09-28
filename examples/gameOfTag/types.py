from enum import Enum


class AgentType(Enum):
    PREDATOR = "predator"
    PREY = "prey"


class Mode(Enum):
    EVALUATE = "evaluate"
    TRAIN = "train"
