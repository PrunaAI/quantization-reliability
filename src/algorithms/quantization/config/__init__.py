from src.algorithms.quantization.config.aqlm_config import AQLM_LORA, AQLM
from src.algorithms.quantization.config.awq_config import AWQ_4
from src.algorithms.quantization.config.base_config import NONE
from src.algorithms.quantization.config.bnb_config import BNB_4, BNB_8
from src.algorithms.quantization.config.hqq_config import HQQ_LORA, HQQ_8_uniform, HQQ_mixed
from src.algorithms.quantization.config.quanto_config import QUANTO, QUANTO_CALIB, QUANTO_QAT

QUANT_CONFIGS = {
    "NONE": NONE,
    "BNB-4": BNB_4,
    "BNB-8": BNB_8,
    "AWQ-4": AWQ_4,
    "HQQ-8-uniform": HQQ_8_uniform,
    "HQQ-mixed": HQQ_mixed,
    "HQQ-LORA": HQQ_LORA,
    "QUANTO": QUANTO,
    "QUANTO-CALIB": QUANTO_CALIB,
    "QUANTO-QAT": QUANTO_QAT,
    "AQLM": AQLM,
    "AQLM-LORA": AQLM_LORA,
}