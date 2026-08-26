from src.model_loading.common.model_enums import QuantizationMethod
from src.model_loading.loaders.aqlm import AQLMModelLoader
from src.model_loading.loaders.awq import AWQModelLoader
from src.model_loading.loaders.bitsandbytes import BitsAndBytesModelLoader
from src.model_loading.loaders.efficient_qat import EfficientQATModelLoader
from src.model_loading.loaders.gptq import GPTQModelLoader
from src.model_loading.loaders.hqq import HQQModelLoader
from src.model_loading.loaders.interface import ModelLoaderInterface
from src.model_loading.loaders.qoq import QoQModelLoader
from src.model_loading.loaders.quanto import QuantoModelLoader
from src.model_loading.loaders.quip import QuipModelLoader
from src.model_loading.loaders.standard import StandardModelLoader


class ModelLoaderFactory:
    """Factory for creating appropriate model loader instances"""
    @staticmethod
    def create_loader(quant_method: QuantizationMethod) -> ModelLoaderInterface:
        loaders = {
            # Standard model loader
            QuantizationMethod.NONE: StandardModelLoader(),
            QuantizationMethod.SPQR: StandardModelLoader(),
            QuantizationMethod.QUAROT: StandardModelLoader(),
            QuantizationMethod.WANDA: StandardModelLoader(),
            QuantizationMethod.WANDA24: StandardModelLoader(),
            QuantizationMethod.SPARSEGPT: StandardModelLoader(),
            QuantizationMethod.SPARSEGPT24: StandardModelLoader(),

            # Main quantization methods
            QuantizationMethod.QOQ: QoQModelLoader(),
            QuantizationMethod.BNB: BitsAndBytesModelLoader(),
            QuantizationMethod.AWQ: AWQModelLoader(),
            QuantizationMethod.QUANTO: QuantoModelLoader(),
            QuantizationMethod.HQQ: HQQModelLoader(),
            QuantizationMethod.GPTQ: GPTQModelLoader(),
            QuantizationMethod.QUIP: QuipModelLoader(),
            QuantizationMethod.EFFICIENTQAT: EfficientQATModelLoader(),
            
            # AQLM and AQLM_PV use the same loader
            QuantizationMethod.AQLM: AQLMModelLoader(),
            QuantizationMethod.AQLM_PV: AQLMModelLoader(),
        }
        return loaders[quant_method]
