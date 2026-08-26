from enum import Enum


class ModelFamily(Enum):
    """Enum representing different model families"""
    TINYLLAMA = "TinyLlama"
    LLAMA2 = "Llama-2"
    LLAMA3 = "Llama-3"
    LLAMA3_IT = "Llama-3-Instruct"
    LLAMA31 = "Llama-3.1"
    LLAMA32 = "Llama-3.2"
    LLAMA32_IT = "Llama-3.2-Instruct"
    BLOOMZ = "Bloomz"
    GPT2 = "GPT2"
    OPT = "OPT"
    QWEN25VL = "Qwen2.5-VL"
    QWEN3 = "Qwen3"
    QWEN3VL = "Qwen3-VL"
    
class ModelSize(Enum):
    """Enum representing model sizes with actual parameter counts"""
    M125 = "125M"    # Opt 125M size
    M350 = "350M"    # Opt 350M size
    B1 = "1B"      # TinyLlama and Llama-3.2 tiny size
    B1_3 = "1.3B"   # Opt 1.3B size
    B1_7 = "1.7B"   # Qwen3 1.7B size
    B2 = "2B"      # Qwen3-VL 2B size
    B2_7 = "2.7B"   # Opt 2.7B size
    B3 = "3B"      # Llama-3.2 small size
    B4 = "4B"      # Qwen3 4B size
    B6_7 = "6.7B"   # Opt 6.7B size
    B7 = "7b"      # Llama-2 size
    B8 = "8B"      # Llama-3 and Llama-3.2 medium size and Qwen3 8B size
    B13 = "13B"    # Opt 13B size
    B14 = "14B"    # Qwen3 14B size
    B30 = "30B"    # Opt 30B size
    B32 = "32B"    # Qwen3 32B size
    B66 = "66B"    # Opt 66B size
    B70 = "70B"    # Llama-3 large size
    B72 = "72B"    # Qwen2.5-VL size

    @classmethod
    def from_size(cls, size_str: str) -> 'ModelSize':
        """Get enum member from size string (e.g., '1B' -> B1)"""
        size_map = {
            "125M": cls.M125,
            "350M": cls.M350,
            "1B": cls.B1,
            "1.3B": cls.B1_3,
            "1.7B": cls.B1_7,
            "2B": cls.B2,
            "2.7B": cls.B2_7,
            "3B": cls.B3,
            "4B": cls.B4,
            "6.7B": cls.B6_7,
            "7B": cls.B7,
            "8B": cls.B8,
            "13B": cls.B13,
            "30B": cls.B30,
            "32B": cls.B32,
            "66B": cls.B66,
            "70B": cls.B70,
            "72B": cls.B72
        }
        return size_map.get(size_str)

    def to_string(self) -> str:
        """Convert to string format for model names (e.g., B1 -> '1b')"""
        string_map = {
            self.M125: "125m",
            self.M350: "350m",
            self.B1: "1b",
            self.B1_3: "1.3b",
            self.B1_7: "1.7b",
            self.B2: "2b",
            self.B2_7: "2.7b",
            self.B3: "3b",
            self.B4: "4b",
            self.B6_7: "6.7b",
            self.B7: "7b",
            self.B8: "8b",
            self.B13: "13b",
            self.B30: "30b",
            self.B32: "32b",
            self.B66: "66b",
            self.B70: "70b",
            self.B72: "72b"
        }
        return string_map[self]

class QuantizationMethod(Enum):
    """Enum representing different quantization methods"""
    NONE = "NONE"
    AWQ = "AWQ"
    GPTQ = "GPTQ"
    AQLM = "AQLM"
    AQLM_PV = "AQLM-PV"
    HQQ = "HQQ"
    QUANTO = "QUANTO"
    BNB = "BNB"
    SPQR = "SPQR"
    QUAROT = "QUAROT"
    QOQ = "QOQ"
    QUIP = "QUIP"
    WANDA = "WANDA"
    WANDA24 = "WANDA24"
    SPARSEGPT = "SPARSEGPT"
    SPARSEGPT24 = "SPARSEGPT24"
    EFFICIENTQAT = "EFFICIENTQAT"

class BitPrecision(Enum):
    """Enum representing different bit precisions for quantization"""
    FP32 = 32
    FP16 = 16
    INT8 = 8
    INT4 = 4
    INT3 = 3
    INT2 = 2
    INT1 = 1
    MIXED = 0  # For mixed precision
