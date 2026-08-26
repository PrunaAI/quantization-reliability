from dataclasses import dataclass
from typing import Optional
from src.model_loading.common.model_enums import BitPrecision, ModelFamily, ModelSize, QuantizationMethod

@dataclass(frozen=True)  # To fix the "unhashable type" error
class ModelIdentifier:
    """Dataclass for unique model identification"""
    family: ModelFamily
    size: ModelSize
    is_chat: bool = False
    quantization: Optional[QuantizationMethod] = None
    bits: Optional[BitPrecision] = None
    is_local: bool = False
    
    # Class method to create instances from string components
    @classmethod
    def from_components(cls, 
                        family: ModelFamily,
                        size: ModelSize,
                        is_chat: bool = False,
                        quantization: Optional[QuantizationMethod] = None,
                        bits: Optional[BitPrecision] = None,
                        is_local: bool = False):
        """Factory method to create a ModelIdentifier from components"""
        return cls(
            family=family,
            size=size,
            is_chat=is_chat,
            quantization=quantization,
            bits=bits,
            is_local=is_local
        )
    
    def __str__(self) -> str:
        """Generate standardized model name string"""
        from src.model_loading.registry.string_utils import ModelStringifier
        return ModelStringifier.to_string(self)
