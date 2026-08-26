from typing import Dict
from src.model_loading.common.model_enums import BitPrecision, ModelFamily, ModelSize, QuantizationMethod
from src.model_loading.common.identifier import ModelIdentifier

class BaseModelConfigs:
    """Base configurations for non-quantized models"""
    BLOOMZ = ModelIdentifier(
        family=ModelFamily.BLOOMZ,
        size=ModelSize.B1
    )
    
    GPT2_LARGE = ModelIdentifier(
        family=ModelFamily.GPT2,
        size=ModelSize.B8
    )

class TinyLlamaConfigs:
    """Configurations for TinyLlama model variants"""
    class Base:
        """Base TinyLlama models"""
        CHAT = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1,
            is_chat=True
        )
        BASE = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized TinyLlama models"""
        BNB_8BIT = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1,
            quantization=QuantizationMethod.BNB,
            bits=BitPrecision.INT8
        )
        BNB_4BIT = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1,
            quantization=QuantizationMethod.BNB,
            bits=BitPrecision.INT4
        )
        AWQ_4BIT = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1,
            quantization=QuantizationMethod.AWQ,
            bits=BitPrecision.INT4
        )
        GPTQ_8BIT = ModelIdentifier(
            family=ModelFamily.TINYLLAMA,
            size=ModelSize.B1,
            quantization=QuantizationMethod.GPTQ,
            bits=BitPrecision.INT8
        )
    
    class Local:
        """Locally quantized TinyLlama models"""
        

class Llama3Configs:
    """Configurations for Llama-3 model variants"""
    class Base:
        """Base Llama-3 models"""
        LLAMA_3_8B = ModelIdentifier(
            family=ModelFamily.LLAMA3,
            size=ModelSize.B8
        )
        LLAMA_3_70B = ModelIdentifier(
            family=ModelFamily.LLAMA3,
            size=ModelSize.B70
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-3 models"""
        class EightB:
            """8B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM models
            AQLM_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM,
                bits=BitPrecision.INT2
            )
            
            # AQLM-PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            AQLM_PV_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT1
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # QoQ models
            QOQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QOQ,
                bits=BitPrecision.INT4
            )
            
            # QuaRot models
            QUAROT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUAROT,
                bits=BitPrecision.INT8
            )
            QUAROT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUAROT,
                bits=BitPrecision.INT4
            )
            
            # QuIP models
            QUIP_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUIP,
                bits=BitPrecision.INT2
            )
        
        class SeventyB:
            """70B model variants"""
            # # GPTQ HF model
            # GPTQ_4BIT = ModelIdentifier(
            #     family=ModelFamily.LLAMA3,
            #     size=ModelSize.B70,
            #     quantization=QuantizationMethod.GPTQ,
            #     bits=BitPrecision.INT4
            # )
            
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM models
            AQLM_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM,
                bits=BitPrecision.INT2
            )
            
            # AQLM-PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            AQLM_PV_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT1
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
    
    class Local:
        """Locally quantized Llama-3 models"""
        class EightB:
            """8B model variants"""
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8,
                is_local=True
            )
            QUANTO_8BIT_CALIB = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8,
                is_local=True
            )
            QUANTO_MIXED = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.MIXED,
                is_local=True
            )
            
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )
            AWQ_4BIT_OASST = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True,
                is_chat=True
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8,
                is_local=True
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4,
                is_local=True
            )
            
            # HQQ models
            HQQ_8BIT_UNIFORM = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8,
                is_local=True
            )
            HQQ_MIXED = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.MIXED,
                is_local=True
            )
            HQQ_LORA = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.MIXED,
                is_local=True
            )
            
            # Wanda pruning
            WANDA_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.WANDA24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT4,
                is_local=True
            )
            WANDA_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
            # SparseGPT pruning
            SPARSEGPT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.SPARSEGPT24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT4,
                is_local=True
            )
            SPARSEGPT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
        class SeventyB:
            """70B model variants"""
            # Wanda pruning
            WANDA_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.WANDA24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT4,
                is_local=True
            )
            WANDA_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
            # SparseGPT pruning
            SPARSEGPT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.SPARSEGPT24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT4,
                is_local=True
            )
            SPARSEGPT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3,
                size=ModelSize.B70,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT2,
                is_local=True
            )

class Llama31Configs:
    """Configurations for Llama-3.1 model variants"""
    class Base:
        """Base Llama-3.1 models"""
        LLAMA_31_8B = ModelIdentifier(
            family=ModelFamily.LLAMA31,
            size=ModelSize.B8
        )
        LLAMA_31_70B = ModelIdentifier(
            family=ModelFamily.LLAMA31,
            size=ModelSize.B70
        )
        
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-3.1 models"""
        class EightB:
            """8B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # AQLM PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            AQLM_PV_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT1
            )
            
    class Local:
        """Locally quantized Llama-3.1 models"""
        class EightB:
            """8B model variants"""
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA31,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )

class Llama3InstructConfigs:
    """Configurations for Llama-3 Instruct model variants"""
    class Base:
        """Base Llama-3 Instruct models"""
        LLAMA_3_8B = ModelIdentifier(
            family=ModelFamily.LLAMA3_IT,
            size=ModelSize.B8
        )
        LLAMA_3_70B = ModelIdentifier(
            family=ModelFamily.LLAMA3_IT,
            size=ModelSize.B70
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-3 Instruct models"""
        class EightB:
            """8B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM models
            AQLM_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM,
                bits=BitPrecision.INT2
            )
            
            # AQLM-PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            AQLM_PV_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT1
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # QoQ models
            QOQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QOQ,
                bits=BitPrecision.INT4
            )
            
            # QuaRot models
            QUAROT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUAROT,
                bits=BitPrecision.INT8
            )
            QUAROT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUAROT,
                bits=BitPrecision.INT4
            )
            
            # QuIP models
            QUIP_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUIP,
                bits=BitPrecision.INT4
            )
        
        class SeventyB:
            """70B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM models
            AQLM_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM,
                bits=BitPrecision.INT2
            )
            
            # AQLM-PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            AQLM_PV_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT1
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B70,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
    
    class Local:
        """Locally quantized Llama-3 Instruct models"""
        class EightB:
            """8B model variants"""
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8,
                is_local=True
            )
            QUANTO_8BIT_CALIB = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8,
                is_local=True
            )
            QUANTO_MIXED = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.MIXED,
                is_local=True
            )
            
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )
            AWQ_4BIT_OASST = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True,
                is_chat=True
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8,
                is_local=True
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4,
                is_local=True
            )
            
            # HQQ models
            HQQ_8BIT_UNIFORM = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8,
                is_local=True
            )
            HQQ_MIXED = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.MIXED,
                is_local=True
            )
            HQQ_LORA = ModelIdentifier(
                family=ModelFamily.LLAMA3_IT,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.MIXED,
                is_local=True
            )

class Llama32InstructConfigs:
    """Configurations for Llama-3.2 Instruct model variants"""
    class Base:
        """Base Llama-3.2 Instruct models"""
        LLAMA_32_1B = ModelIdentifier(
            family=ModelFamily.LLAMA32_IT,
            size=ModelSize.B1
        )
        LLAMA_32_3B = ModelIdentifier(
            family=ModelFamily.LLAMA32_IT,
            size=ModelSize.B3
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-3.2 Instruct models"""
        class OneB:
            """1B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # Quanto models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
            
            # QuIP models
            QUIP_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUIP,
                bits=BitPrecision.INT4
            )
        
        class ThreeB:
            """3B model variants"""
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # AQLM PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
    
    class Local:
        """Locally quantized Llama-3.2 Instruct models"""
        class OneB:
            """1B model variants"""
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )
        
        class ThreeB:
            """3B model variants"""
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32_IT,
                size=ModelSize.B3,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )

class Llama32Configs:
    """Configurations for Llama-3.2 model variants"""
    class Base:
        """Base Llama-3.2 models"""
        LLAMA_32_1B = ModelIdentifier(
            family=ModelFamily.LLAMA32,
            size=ModelSize.B1
        )
        LLAMA_32_3B = ModelIdentifier(
            family=ModelFamily.LLAMA32,
            size=ModelSize.B3
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-3.2 models"""
        class OneB:
            """1B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # Quanto models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # AQLM PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
        
        class ThreeB:
            """3B model variants"""
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            # AQLM PV models
            AQLM_PV_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.AQLM_PV,
                bits=BitPrecision.INT2
            )
    
    class Local:
        """Locally quantized Llama-3.2 models"""
        class OneB:
            """1B model variants"""
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )
            
            # Wanda pruning
            WANDA_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.WANDA24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT4,
                is_local=True
            )
            WANDA_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
            # SparseGPT pruning
            SPARSEGPT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.SPARSEGPT24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT4,
                is_local=True
            )
            SPARSEGPT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B1,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT2,
                is_local=True
            )
        
        class ThreeB:
            """3B model variants"""
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4,
                is_local=True
            )
            
            # Wanda pruning
            WANDA_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.WANDA24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            WANDA_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT4,
                is_local=True
            )
            WANDA_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.WANDA,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
            # SparseGPT pruning
            SPARSEGPT_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT24_8BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.SPARSEGPT24,
                bits=BitPrecision.INT8,
                is_local=True
            )
            SPARSEGPT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT4,
                is_local=True
            )
            SPARSEGPT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA32,
                size=ModelSize.B3,
                quantization=QuantizationMethod.SPARSEGPT,
                bits=BitPrecision.INT2,
                is_local=True
            )
            
class Llama2Configs:
    """Configurations for Llama-2 model variants"""
    class Base:
        """Base Llama-2 models"""
        LLAMA_2_7B = ModelIdentifier(
            family=ModelFamily.LLAMA2,
            size=ModelSize.B7
        )
        LLAMA_2_13B = ModelIdentifier(
            family=ModelFamily.LLAMA2,
            size=ModelSize.B13
        )
        LLAMA_2_70B = ModelIdentifier(
            family=ModelFamily.LLAMA2,
            size=ModelSize.B70
        )
        
    class HuggingFace:
        """HuggingFace-hosted quantized Llama-2 models"""
        class SevenB:
            """7B model variants"""
            # EFFICIENTQAT models
            EFFICIENTQAT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B7,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT4
            )
            EFFICIENTQAT_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B7,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT3
            )
            EFFICIENTQAT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B7,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT2
            )
            
        class ThirteenB:
            """13B model variants"""
            # EFFICIENTQAT models
            EFFICIENTQAT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B13,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT4
            )
            EFFICIENTQAT_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B13,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT3
            )
            EFFICIENTQAT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B13,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT2
            )
            
        class SeventyB:
            """70B model variants"""
            # EFFICIENTQAT models
            EFFICIENTQAT_4BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B70,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT4
            )
            EFFICIENTQAT_3BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B70,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT3
            )
            EFFICIENTQAT_2BIT = ModelIdentifier(
                family=ModelFamily.LLAMA2,
                size=ModelSize.B70,
                quantization=QuantizationMethod.EFFICIENTQAT,
                bits=BitPrecision.INT2
            )
        
class OptConfigs:
    """Configurations for OPT model variants"""
    class Base:
        """Base OPT models"""
        OPT_125M = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.M125
        )
        OPT_350M = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.M350
        )
        OPT_1B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B1_3
        )
        OPT_2B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B2_7
        )
        OPT_6B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B6_7
        )
        OPT_13B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B13
        )
        OPT_30B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B30
        )
        OPT_66B = ModelIdentifier(
            family=ModelFamily.OPT,
            size=ModelSize.B66
        )
    
    class HuggingFace:
        """HuggingFace-hosted quantized OPT models"""
        class M125:
            """125M model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.OPT,
            #     size=ModelSize.M125,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M125,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class M350:
            """350M model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.OPT,
            #     size=ModelSize.M350,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.M350,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
        class B1_3:
            """1.3B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            HQQ_1BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT1
            )
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B1_3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B2_7:
            """2.7B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B2_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
        
        class B6_7:
            """6.7B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B6_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B13:
            """13B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B13,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B30:
            """30B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B30,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B66:
            """66B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.OPT,
                size=ModelSize.B66,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
            
class Qwen25VLConfigs:
    """Configurations for Qwen-2.5-VL model variants"""
    class Base:
        """Base Qwen-2.5-VL models"""
        QWEN_2_5_VL_3B = ModelIdentifier(
            family=ModelFamily.QWEN25VL,
            size=ModelSize.B3
        )
        QWEN_2_5_VL_7B = ModelIdentifier(
            family=ModelFamily.QWEN25VL,
            size=ModelSize.B7
        )
        QWEN_2_5_VL_32B = ModelIdentifier(
            family=ModelFamily.QWEN25VL,
            size=ModelSize.B32
        )
        QWEN_2_5_VL_72B = ModelIdentifier(
            family=ModelFamily.QWEN25VL,
            size=ModelSize.B72
        )
        
    class HuggingFace:
        """HuggingFace-hosted quantized Qwen-2.5-VL models"""
        class B3:
            """3B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN25VL,
            #     size=ModelSize.B3,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B3,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B7:
            """7B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN25VL,
            #     size=ModelSize.B7,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B32:
            """32B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN25VL,
            #     size=ModelSize.B32,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B72:
            """72B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN25VL,
            #     size=ModelSize.B72,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN25VL,
                size=ModelSize.B72,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
class Qwen3VLConfigs:
    """Configurations for Qwen-3-VL model variants"""
    class Base:
        """Base Qwen-3-VL models"""
        QWEN_3_VL_2B = ModelIdentifier(
            family=ModelFamily.QWEN3VL,
            size=ModelSize.B2
        )
        QWEN_3_VL_4B = ModelIdentifier(
            family=ModelFamily.QWEN3VL,
            size=ModelSize.B4
        )
        QWEN_3_VL_8B = ModelIdentifier(
            family=ModelFamily.QWEN3VL,
            size=ModelSize.B8
        )
        QWEN_3_VL_32B = ModelIdentifier(
            family=ModelFamily.QWEN3VL,
            size=ModelSize.B32
        )
        
    class HuggingFace:
        """HuggingFace-hosted quantized Qwen-3-VL models"""
        class B2:
            """2B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3VL,
            #     size=ModelSize.B2,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B2,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B4:
            """4B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3VL,
            #     size=ModelSize.B4,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B8:
            """8B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3VL,
            #     size=ModelSize.B8,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B32:
            """32B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3VL,
            #     size=ModelSize.B32,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3VL,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )


class Qwen3Configs:
    """Configurations for Qwen-3 model variants"""
    class Base:
        """Base Qwen-3 models"""
        QWEN_3_1B = ModelIdentifier(
            family=ModelFamily.QWEN3,
            size=ModelSize.B1_7
        )
        QWEN_3_4B = ModelIdentifier(
            family=ModelFamily.QWEN3,
            size=ModelSize.B4
        )
        QWEN_3_8B = ModelIdentifier(
            family=ModelFamily.QWEN3,
            size=ModelSize.B8
        )
        QWEN_3_14B = ModelIdentifier(
            family=ModelFamily.QWEN3,
            size=ModelSize.B14
        )
        QWEN_3_32B = ModelIdentifier(
            family=ModelFamily.QWEN3,
            size=ModelSize.B32
        )
        
    class HuggingFace:
        """HuggingFace-hosted quantized Qwen-3 models"""
        class B1_7:
            """1.7B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3,
            #     size=ModelSize.B1_7,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B1_7,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B4:
            """4B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8,
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3,
            #     size=ModelSize.B4,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B4,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B8:
            """8B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3,
            #     size=ModelSize.B8,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B8,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B14:
            """14B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3,
            #     size=ModelSize.B14,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B14,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )
            
        class B32:
            """32B model variants"""
            # AWQ models
            AWQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.AWQ,
                bits=BitPrecision.INT4
            )
            
            # BNB models
            BNB_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT8
            )
            BNB_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.BNB,
                bits=BitPrecision.INT4
            )
            
            # HQQ models
            HQQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT8
            )
            HQQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT4
            )
            HQQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT3
            )
            HQQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.HQQ,
                bits=BitPrecision.INT2
            )
            # HQQ_1BIT = ModelIdentifier(
            #     family=ModelFamily.QWEN3,
            #     size=ModelSize.B32,
            #     quantization=QuantizationMethod.HQQ,
            #     bits=BitPrecision.INT1
            # )
            
            # QUANTO models
            QUANTO_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT8
            )
            QUANTO_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT4
            )
            QUANTO_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.QUANTO,
                bits=BitPrecision.INT2
            )
            
            # GPTQ models
            GPTQ_8BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT8
            )
            GPTQ_4BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT4
            )
            GPTQ_3BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT3
            )
            GPTQ_2BIT = ModelIdentifier(
                family=ModelFamily.QWEN3,
                size=ModelSize.B32,
                quantization=QuantizationMethod.GPTQ,
                bits=BitPrecision.INT2
            )

class Models:
    """Central registry of model configurations"""
    BaseModels = BaseModelConfigs
    TinyLlama = TinyLlamaConfigs
    Llama2 = Llama2Configs
    Llama3 = Llama3Configs
    Llama3Instruct = Llama3InstructConfigs
    Llama31 = Llama31Configs
    Llama32 = Llama32Configs
    Llama32Instruct = Llama32InstructConfigs
    Opt = OptConfigs
    Qwen25VL = Qwen25VLConfigs
    Qwen3 = Qwen3Configs
    Qwen3VL = Qwen3VLConfigs

    @classmethod
    def get_all_models(cls) -> Dict[str, ModelIdentifier]:
        """Get all model configurations as a dictionary"""
        def process_class(class_obj, prefix=""):
            models = {}
            for name, value in vars(class_obj).items():
                if name.startswith('_'):
                    continue
                if isinstance(value, type):
                    nested_models = process_class(value, f"{prefix}{name}_")
                    models.update(nested_models)
                else:
                    models[f"{prefix}{name}"] = value
            return models

        all_models = {}
        for config_class in [
            cls.BaseModels,
            cls.TinyLlama,
            cls.Llama2,
            cls.Llama3,
            cls.Llama3Instruct,
            cls.Llama31,
            cls.Llama32,
            cls.Llama32Instruct,
            cls.Opt,
            cls.Qwen25VL,
            cls.Qwen3,
            cls.Qwen3VL
        ]:
            models = process_class(config_class, f"{config_class.__name__}_")
            all_models.update(models)
                    
        return all_models