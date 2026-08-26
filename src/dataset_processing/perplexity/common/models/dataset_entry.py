from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerplexityDatasetEntry:
    """Single entry in a perplexity dataset."""
    input_ids: Any
    target_ids: Any
    metadata: Dict[str, Any]