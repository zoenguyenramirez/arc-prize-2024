from enum import Enum, auto

class SpecialToken(Enum):
    CELL_TOKEN_SIZE = 10
    PAD = auto()                # 11
    START = auto()              # 12
    START_INPUT = auto()        # 13
    END_INPUT = auto()          # 14
    START_OUTPUT = auto()       # 15
    END_OUTPUT = auto()         # 16
    END = auto()                # 17
    ROW_SEPARATOR = auto()      # 18
    COUNT_OF_TOKENS = auto()    # 19


VOCAB_SIZE = SpecialToken.COUNT_OF_TOKENS.value
