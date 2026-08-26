from typing import Optional
from src.model_loading.common.model_enums import BitPrecision, ModelFamily, ModelSize, QuantizationMethod
from src.model_loading.common.identifier import ModelIdentifier


_FAMILY_STR = {
    ModelFamily.LLAMA3: "llama3",
    ModelFamily.LLAMA2: "llama2",
    ModelFamily.LLAMA32: "llama32",
    ModelFamily.LLAMA31: "llama31",
    ModelFamily.OPT: "opt",
    ModelFamily.QWEN25VL: "qwen25vl",
    ModelFamily.QWEN3: "qwen3",
    ModelFamily.QWEN3VL: "qwen3vl",
    ModelFamily.LLAMA3_IT: "llama3_it",
    ModelFamily.LLAMA32_IT: "llama32_it",
}

_SIZE_STR = {
    ModelSize.M125: "125m",
    ModelSize.M350: "350m",
    ModelSize.B1_3: "1.3b",
    ModelSize.B1_7: "1.7b",
    ModelSize.B2: "2b",
    ModelSize.B2_7: "2.7b",
    ModelSize.B4: "4b",
    ModelSize.B6_7: "6.7b",
    ModelSize.B13: "13b",
    ModelSize.B14: "14b",
    ModelSize.B1: "1b",
    ModelSize.B3: "3b",
    ModelSize.B7: "7b",
    ModelSize.B8: "8b",
    ModelSize.B30: "30b",
    ModelSize.B32: "32b",
    ModelSize.B66: "66b",
    ModelSize.B70: "70b",
    ModelSize.B72: "72b",
}


class ModelStringifier:
    """Handles conversion between ModelIdentifiers and their string representations"""

    @staticmethod
    def to_string(model: ModelIdentifier) -> str:
        """Convert a ModelIdentifier to its string representation"""
        components = []

        # Add model family
        family_str = _FAMILY_STR.get(model.family, model.family.name.lower())
        components.append(family_str)

        # Add size
        size_str = _SIZE_STR.get(model.size)
        if size_str:
            components.append(size_str)

        # Add chat indicator
        if model.is_chat:
            components.append("chat")
            
        # Add quantization method and bits
        if model.quantization:
            if model.quantization == QuantizationMethod.AQLM_PV:
                components.append("aqlm-pv")  # Use hyphen instead of underscore
            else:
                components.append(model.quantization.name.lower())
                
            if model.bits:
                components.append(f"{model.bits.value}bit")
                
        # Add local indicator
        if model.is_local:
            components.append("local")
            
        return "_".join(components)

    @staticmethod
    def from_string(model_str: str) -> Optional[ModelIdentifier]:
        """Convert a string representation to a ModelIdentifier using factory method"""
        # Pre-process: Replace "aqlm_pv" with "aqlm-pv" to avoid splitting issues
        if "aqlm_pv" in model_str.lower():
            model_str = model_str.lower().replace("aqlm_pv", "aqlm-pv")
        
        components = model_str.lower().split("_")
        
        # Check for instruct variant pattern and normalize it
        if len(components) > 1 and components[1] == "it":
            # Handle pattern like "llama3_it_8b_..."
            components = [components[0] + "_it"] + components[2:]
        
        # Parse model family
        family_map = {
            "tinyllama": ModelFamily.TINYLLAMA,
            "llama2": ModelFamily.LLAMA2,
            "llama3": ModelFamily.LLAMA3,
            "llama31": ModelFamily.LLAMA31,
            "llama32": ModelFamily.LLAMA32,
            "llama3_it": ModelFamily.LLAMA3_IT,
            "llama32_it": ModelFamily.LLAMA32_IT,
            "bloomz": ModelFamily.BLOOMZ,
            "gpt2": ModelFamily.GPT2,
            "opt": ModelFamily.OPT,
            "qwen25vl": ModelFamily.QWEN25VL,
            "qwen3vl": ModelFamily.QWEN3VL,
            "qwen3": ModelFamily.QWEN3,
        }
        family = family_map.get(components[0])
        if not family:
            return None
            
        # Parse size
        size_map = {
            "125m": ModelSize.M125,
            "350m": ModelSize.M350,
            "1.3b": ModelSize.B1_3,
            "1.7b": ModelSize.B1_7,
            "2b": ModelSize.B2,
            "2.7b": ModelSize.B2_7,
            "6.7b": ModelSize.B6_7,
            "13b": ModelSize.B13,
            "1b": ModelSize.B1,
            "3b": ModelSize.B3,
            "4b": ModelSize.B4,
            "7b": ModelSize.B7,
            "8b": ModelSize.B8,
            "30b": ModelSize.B30,
            "32b": ModelSize.B32,
            "66b": ModelSize.B66,
            "70b": ModelSize.B70,
            "72b": ModelSize.B72
        }
        size = None
        for component in components[1:]:
            if component in size_map:
                size = size_map[component]
                break
        if not size:
            return None
            
        # Parse optional fields
        is_chat = "chat" in components
        is_local = "local" in components
        
        # Parse quantization method
        quant_map = {
            "bnb": QuantizationMethod.BNB,
            "awq": QuantizationMethod.AWQ,
            "gptq": QuantizationMethod.GPTQ,
            "hqq": QuantizationMethod.HQQ,
            "quanto": QuantizationMethod.QUANTO,
            "aqlm-pv": QuantizationMethod.AQLM_PV,
            "aqlm": QuantizationMethod.AQLM,
            "qoq": QuantizationMethod.QOQ,
            "quarot": QuantizationMethod.QUAROT,
            "quip": QuantizationMethod.QUIP,
            "wanda": QuantizationMethod.WANDA,
            "wanda24": QuantizationMethod.WANDA24,
            "sparsegpt": QuantizationMethod.SPARSEGPT,
            "sparsegpt24": QuantizationMethod.SPARSEGPT24,
            "efficientqat": QuantizationMethod.EFFICIENTQAT
        }
        quantization = None
        for component in components:
            if component in quant_map:
                quantization = quant_map[component]
                break
                
        # Parse bit precision
        bits = None
        for component in components:
            if component.endswith("bit"):
                try:
                    bit_value = int(component[:-3])
                    bit_map = {
                        1: BitPrecision.INT1,
                        2: BitPrecision.INT2,
                        3: BitPrecision.INT3,
                        4: BitPrecision.INT4,
                        8: BitPrecision.INT8
                    }
                    bits = bit_map.get(bit_value)
                except ValueError:
                    pass
        
        # Use the factory method to create the ModelIdentifier
        return ModelIdentifier.from_components(
            family=family,
            size=size,
            is_chat=is_chat,
            quantization=quantization,
            bits=bits,
            is_local=is_local
        )
