from enum import Enum

class PerplexityDatasetType(Enum):
    WIKITEXT = "WIKITEXT"
    PTB = "PTB"
    C4 = "C4"
    OPENASSISTANT = "OPENASSISTANT"
    POLYGLOT = "POLYGLOT"