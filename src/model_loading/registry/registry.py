import os
from pathlib import Path
from typing import Dict, Optional
from src.config import MODEL_SAVE_DIR, MODEL_CACHE_ROOT
from src.model_loading.common.model_enums import BitPrecision
from src.model_loading.common.identifier import ModelIdentifier
from src.model_loading.common.paths import ModelPaths
from src.model_loading.registry.models import Models

class ModelRegistry:
    """Registry for model paths and configurations"""
    def __init__(self, model_save_path: Optional[str] = None):
        self.model_save_path = Path(model_save_path) if model_save_path else Path(MODEL_SAVE_DIR)
        self.ceph_model_path = Path(MODEL_CACHE_ROOT)
        self._model_paths: Dict[ModelIdentifier, ModelPaths] = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all model configurations"""
        self._initialize_base_models()
        self._initialize_tinyllama_models()
        self._initialize_llama2_base_models()
        self._initialize_llama2_huggingface_models()
        self._initialize_llama3_models()
        self._initialize_llama31_models()
        self._initialize_llama32_models()
        self._initialize_opt_base_models()
        self._initialize_opt_huggingface_models()
        self._initialize_qwen25vl_base_models()
        self._initialize_qwen25vl_huggingface_models()
        self._initialize_qwen3vl_base_models()
        self._initialize_qwen3vl_huggingface_models()
        self._initialize_qwen3_base_models()
        self._initialize_qwen3_huggingface_models()
        
    def _initialize_base_models(self):
        """Initialize all base model configurations"""
        self._register_model(Models.BaseModels.BLOOMZ, "bigscience/bloomz-1b1")
        self._register_model(Models.BaseModels.GPT2_LARGE, "openai-community/gpt2-large")
        
    def _initialize_tinyllama_models(self):
        """Initialize all TinyLlama model configurations"""
        # Base Models
        self._register_model(Models.TinyLlama.Base.CHAT, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self._register_model(Models.TinyLlama.Base.BASE, "TinyLlama/TinyLlama_v1.1")
        
        # HuggingFace Models
        tinyllama_base_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._register_model(
            Models.TinyLlama.HuggingFace.BNB_8BIT,
            "PrunaAI/TinyLlama-TinyLlama-1.1B-Chat-v0.3-bnb-8bit-smashed",
            tinyllama_base_path
        )
        self._register_model(
            Models.TinyLlama.HuggingFace.BNB_4BIT,
            "PrunaAI/TinyLlama-TinyLlama-1.1B-Chat-v0.3-bnb-4bit-smashed",
            tinyllama_base_path
        )
        self._register_model(
            Models.TinyLlama.HuggingFace.AWQ_4BIT,
            "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
            tinyllama_base_path
        )
        self._register_model(
            Models.TinyLlama.HuggingFace.GPTQ_8BIT,
            "PrunaAI/TinyLlama-TinyLlama-1.1B-Chat-v0.3-GPTQ-8bit-smashed",
            tinyllama_base_path
        )

    def _initialize_llama2_base_models(self):
        """Initialize Llama-2 models"""
        # Base Models
        self._register_model(
            Models.Llama2.Base.LLAMA_2_7B,
            "meta-llama/Llama-2-7b-hf"
        )
        self._register_model(
            Models.Llama2.Base.LLAMA_2_13B,
            "meta-llama/Llama-2-13b-hf"
        )
        self._register_model(
            Models.Llama2.Base.LLAMA_2_70B,
            "meta-llama/Llama-2-70b-hf"
        )
        
    def _initialize_llama2_huggingface_models(self):
        """Initialize Llama-2 HuggingFace models"""
        # EFFICIENTQAT models
        self._register_model(
            Models.Llama2.HuggingFace.SevenB.EFFICIENTQAT_4BIT,
            "ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ",
            "ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.SevenB.EFFICIENTQAT_3BIT,
            "ChenMnZ/Llama-2-7b-EfficientQAT-w3g128-GPTQ",
            "ChenMnZ/Llama-2-7b-EfficientQAT-w3g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.SevenB.EFFICIENTQAT_2BIT,
            "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ",
            "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.ThirteenB.EFFICIENTQAT_4BIT,
            "ChenMnZ/Llama-2-13b-EfficientQAT-w4g128-GPTQ",
            "ChenMnZ/Llama-2-13b-EfficientQAT-w4g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.ThirteenB.EFFICIENTQAT_3BIT,
            "ChenMnZ/Llama-2-13b-EfficientQAT-w3g128-GPTQ",
            "ChenMnZ/Llama-2-13b-EfficientQAT-w3g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.ThirteenB.EFFICIENTQAT_2BIT,
            "ChenMnZ/Llama-2-13b-EfficientQAT-w2g128-GPTQ",
            "ChenMnZ/Llama-2-13b-EfficientQAT-w2g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.SeventyB.EFFICIENTQAT_4BIT,
            "ChenMnZ/Llama-2-70b-EfficientQAT-w4g128-GPTQ",
            "ChenMnZ/Llama-2-70b-EfficientQAT-w4g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.SeventyB.EFFICIENTQAT_3BIT,
            "ChenMnZ/Llama-2-70b-EfficientQAT-w3g128-GPTQ",
            "ChenMnZ/Llama-2-70b-EfficientQAT-w3g128-GPTQ"
        )
        self._register_model(
            Models.Llama2.HuggingFace.SeventyB.EFFICIENTQAT_2BIT,
            "ChenMnZ/Llama-2-70b-EfficientQAT-w2g128-GPTQ",
            "ChenMnZ/Llama-2-70b-EfficientQAT-w2g128-GPTQ"
        )

    def _initialize_llama3_models(self):
        """Initialize Llama-3 models"""
        # Base Llama3 models
        self._initialize_llama3_base_models()
        self._initialize_llama3_huggingface_models()
        self._initialize_llama3_local_models()
        self._initialize_llama3_70b_local_models()
        
        # Instruct Llama3 models
        self._initialize_llama3_it_base_models()
        self._initialize_llama3_it_huggingface_models()
        self._initialize_llama3_it_local_models()

    def _initialize_llama3_base_models(self):
        """Initialize Llama-3 base models"""
        self._register_model(
            Models.Llama3.Base.LLAMA_3_8B,
            "meta-llama/Meta-Llama-3-8B"
        )
        self._register_model(
            Models.Llama3.Base.LLAMA_3_70B,
            "meta-llama/Meta-Llama-3-70B"
        )
        
    def _initialize_llama3_it_base_models(self):
        """Initialize Llama-3 Instruct base models"""
        self._register_model(
            Models.Llama3Instruct.Base.LLAMA_3_8B,
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self._register_model(
            Models.Llama3Instruct.Base.LLAMA_3_70B,
            "meta-llama/Meta-Llama-3-70B-Instruct"
        )
        
    def _initialize_llama3_huggingface_models(self):
        """Initialize Llama-3 HuggingFace models"""
        self._initialize_llama3_huggingface_8b_models()
        self._initialize_llama3_huggingface_70b_models()
        
    def _initialize_llama3_it_huggingface_models(self):
        """Initialize Llama-3 Instruct HuggingFace models"""
        self._initialize_llama3_it_huggingface_8b_models()
        self._initialize_llama3_it_huggingface_70b_models()
    
    def _initialize_llama3_it_huggingface_8b_models(self):
        """Initialize Llama-3 Instruct HuggingFace 8B models"""
        base_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # AQLM models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.AQLM_2BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.AQLM_PV_2BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-PV-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.AQLM_PV_1BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-PV-1Bit-1x16",
            base_path
        )

        # AWQ models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.AWQ_4BIT,
            "PrunaAI/meta-llama-Meta-Llama-3-8B-Instruct-AWQ-4bit-smashed",
            base_path
        )
        
        # BNB models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.EightB.BNB_8BIT,
            Models.Llama3Instruct.HuggingFace.EightB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.EightB.GPTQ_8BIT,
            Models.Llama3Instruct.HuggingFace.EightB.GPTQ_4BIT,
            Models.Llama3Instruct.HuggingFace.EightB.GPTQ_3BIT,
            Models.Llama3Instruct.HuggingFace.EightB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.EightB.HQQ_8BIT,
            Models.Llama3Instruct.HuggingFace.EightB.HQQ_4BIT,
            Models.Llama3Instruct.HuggingFace.EightB.HQQ_3BIT,
            Models.Llama3Instruct.HuggingFace.EightB.HQQ_2BIT,
            Models.Llama3Instruct.HuggingFace.EightB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.EightB.QUANTO_8BIT,
            Models.Llama3Instruct.HuggingFace.EightB.QUANTO_4BIT,
            Models.Llama3Instruct.HuggingFace.EightB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)

        # QoQ models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.QOQ_4BIT,
            "mit-han-lab/Llama-3-8B-Instruct-QServe",
            "mit-han-lab/Llama-3-8B-Instruct-QServe"
        )

        # QuaRot models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.QUAROT_8BIT,
            "AminBH/Llama-3-8B-Instruct-Quarot-wiki-8bit",
            "AminBH/Llama-3-8B-Instruct-Quarot-wiki-8bit"
        )
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.QUAROT_4BIT,
            "AminBH/Llama-3-8B-Instruct-Quarot-wiki-4bit",
            "AminBH/Llama-3-8B-Instruct-Quarot-wiki-4bit"
        )
        
        # QuIP models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.EightB.QUIP_4BIT,
            "nm-testing/Meta-Llama-3-8B-Instruct-quip-w4a16",
            "nm-testing/Meta-Llama-3-8B-Instruct-quip-w4a16"
        )
        
    def _initialize_llama3_it_huggingface_70b_models(self):
        """Initialize Llama-3 Instruct HuggingFace 70B models"""
        base_path = "meta-llama/Meta-Llama-3-70B-Instruct"
        
        # AWQ models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.SeventyB.AWQ_4BIT,
            "TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",
            "TechxGenus/Meta-Llama-3-70B-Instruct-AWQ"
        )
        
        # AQLM models
        self._register_model(
            Models.Llama3Instruct.HuggingFace.SeventyB.AQLM_2BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.HuggingFace.SeventyB.AQLM_PV_2BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-PV-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.HuggingFace.SeventyB.AQLM_PV_1BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-PV-1Bit-1x16",
            base_path
        )
        
        # BNB models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.SeventyB.BNB_8BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.SeventyB.GPTQ_8BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.GPTQ_4BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.GPTQ_3BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.SeventyB.HQQ_8BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.HQQ_4BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.HQQ_3BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.HQQ_2BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama3Instruct.HuggingFace.SeventyB.QUANTO_8BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.QUANTO_4BIT,
            Models.Llama3Instruct.HuggingFace.SeventyB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)
    
    def _initialize_llama3_huggingface_8b_models(self):
        """Initialize Llama-3 HuggingFace 8B models"""
        base_path = "meta-llama/Meta-Llama-3-8B"
        
        # AQLM models
        self._register_model(
            Models.Llama3.HuggingFace.EightB.AQLM_2BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3.HuggingFace.EightB.AQLM_PV_2BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3.HuggingFace.EightB.AQLM_PV_1BIT,
            "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16",
            base_path
        )

        # AWQ models
        self._register_model(
            Models.Llama3.HuggingFace.EightB.AWQ_4BIT,
            "PrunaAI/meta-llama-Meta-Llama-3-8B-AWQ-4bit-smashed",
            base_path
        )
        
        # BNB models
        for model_config in [
            Models.Llama3.HuggingFace.EightB.BNB_8BIT,
            Models.Llama3.HuggingFace.EightB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama3.HuggingFace.EightB.GPTQ_8BIT,
            Models.Llama3.HuggingFace.EightB.GPTQ_4BIT,
            Models.Llama3.HuggingFace.EightB.GPTQ_3BIT,
            Models.Llama3.HuggingFace.EightB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama3.HuggingFace.EightB.HQQ_8BIT,
            Models.Llama3.HuggingFace.EightB.HQQ_4BIT,
            Models.Llama3.HuggingFace.EightB.HQQ_3BIT,
            Models.Llama3.HuggingFace.EightB.HQQ_2BIT,
            Models.Llama3.HuggingFace.EightB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama3.HuggingFace.EightB.QUANTO_8BIT,
            Models.Llama3.HuggingFace.EightB.QUANTO_4BIT,
            Models.Llama3.HuggingFace.EightB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)

        # QoQ models
        self._register_model(
            Models.Llama3.HuggingFace.EightB.QOQ_4BIT,
            "mit-han-lab/Llama-3-8B-QServe",
            "mit-han-lab/Llama-3-8B-QServe"
        )

        # QuaRot models
        self._register_model(
            Models.Llama3.HuggingFace.EightB.QUAROT_8BIT,
            "AminBH/Llama-3-8B-Quarot-wiki-8bit",
            "AminBH/Llama-3-8B-Quarot-wiki-8bit"
        )
        self._register_model(
            Models.Llama3.HuggingFace.EightB.QUAROT_4BIT,
            "AminBH/Llama-3-8B-Quarot-wiki-4bit",
            "AminBH/Llama-3-8B-Quarot-wiki-4bit"
        )
        
        # QuIP models
        self._register_model(
            Models.Llama3.HuggingFace.EightB.QUIP_2BIT,
            "Efficient-ML/LLaMA-3-8B-QuIP-2bit",
            "Efficient-ML/LLaMA-3-8B-QuIP-2bit"
        )
        
    def _initialize_llama3_huggingface_70b_models(self):
        """Initialize Llama-3 HuggingFace 70B models"""
        base_path = "meta-llama/Meta-Llama-3-70B"
        
        # AWQ models
        self._register_model(
            Models.Llama3.HuggingFace.SeventyB.AWQ_4BIT,
            "TechxGenus/Meta-Llama-3-70B-AWQ",
            "TechxGenus/Meta-Llama-3-70B-AWQ"
        )
        
        # # GPTQ models
        # self._register_model(
        #     Models.Llama3.HuggingFace.SeventyB.GPTQ_4BIT,
        #     "TechxGenus/Meta-Llama-3-70B-GPTQ",
        #     "TechxGenus/Meta-Llama-3-70B-GPTQ"
        # )
        
        # AQLM models
        self._register_model(
            Models.Llama3.HuggingFace.SeventyB.AQLM_2BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-AQLM-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3.HuggingFace.SeventyB.AQLM_PV_2BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-2Bit-1x16",
            base_path
        )
        self._register_model(
            Models.Llama3.HuggingFace.SeventyB.AQLM_PV_1BIT,
            "ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-1Bit-1x16",
            base_path
        )
        
        # BNB models
        for model_config in [
            Models.Llama3.HuggingFace.SeventyB.BNB_8BIT,
            Models.Llama3.HuggingFace.SeventyB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama3.HuggingFace.SeventyB.GPTQ_8BIT,
            Models.Llama3.HuggingFace.SeventyB.GPTQ_4BIT,
            Models.Llama3.HuggingFace.SeventyB.GPTQ_3BIT,
            Models.Llama3.HuggingFace.SeventyB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama3.HuggingFace.SeventyB.HQQ_8BIT,
            Models.Llama3.HuggingFace.SeventyB.HQQ_4BIT,
            Models.Llama3.HuggingFace.SeventyB.HQQ_3BIT,
            Models.Llama3.HuggingFace.SeventyB.HQQ_2BIT,
            Models.Llama3.HuggingFace.SeventyB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama3.HuggingFace.SeventyB.QUANTO_8BIT,
            Models.Llama3.HuggingFace.SeventyB.QUANTO_4BIT,
            Models.Llama3.HuggingFace.SeventyB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)
        
    def _initialize_llama3_local_models(self):
        """Initialize Llama-3 local models"""
        base_path = "meta-llama/Meta-Llama-3-8B"
        
        # QUANTO models
        self._register_model(
            Models.Llama3.Local.EightB.QUANTO_8BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-QUANTO-8"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.QUANTO_8BIT_CALIB,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-QUANTO-CALIB-8"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.QUANTO_MIXED,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-QUANTO-8-mixed"),
            base_path
        )

        # AWQ models
        self._register_model(
            Models.Llama3.Local.EightB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-AWQ-4bit"),
            base_path
        )

        # BNB models
        self._register_model(
            Models.Llama3.Local.EightB.BNB_8BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-BNB-8"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.BNB_4BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-BNB-4"),
            base_path
        )

        # HQQ models
        self._register_model(
            Models.Llama3.Local.EightB.HQQ_8BIT_UNIFORM,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-HQQ-8-uniform"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.HQQ_MIXED,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-HQQ-mixed"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.HQQ_LORA,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-HQQ-LORA"),
            base_path
        )
        
        # Wanda models
        self._register_model(
            Models.Llama3.Local.EightB.WANDA_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Meta-Llama-3-8B_wanda_unstructured_0.5"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.WANDA24_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Meta-Llama-3-8B_wanda_2:4_0.5"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.WANDA_4BIT,
            os.path.join(self.ceph_model_path, "wanda", "Meta-Llama-3-8B_wanda_unstructured_0.3"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.WANDA_2BIT,
            os.path.join(self.ceph_model_path, "wanda", "Meta-Llama-3-8B_wanda_unstructured_0.1"),
            base_path
        )
        
        # SparseGPT models
        self._register_model(
            Models.Llama3.Local.EightB.SPARSEGPT_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Meta-Llama-3-8B_sparsegpt_unstructured_0.5"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.SPARSEGPT24_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Meta-Llama-3-8B_sparsegpt_2:4_0.5"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.SPARSEGPT_4BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Meta-Llama-3-8B_sparsegpt_unstructured_0.3"),
            base_path
        )
        self._register_model(
            Models.Llama3.Local.EightB.SPARSEGPT_2BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Meta-Llama-3-8B_sparsegpt_unstructured_0.1"),
            base_path
        )
        
    def _initialize_llama3_70b_local_models(self):
        """Initialize Llama-3 70B local models"""        
        # Wanda models
        self._register_model(
            Models.Llama3.Local.SeventyB.WANDA_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-2-70b-hf_wanda_unstructured_0.5"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.WANDA24_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-2-70b-hf_wanda_2:4_0.5"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.WANDA_4BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-2-70b-hf_wanda_unstructured_0.3"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.WANDA_2BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-2-70b-hf_wanda_unstructured_0.1"),
            "meta-llama/Llama-2-70b-hf"
        )
        
        # SparseGPT models
        self._register_model(
            Models.Llama3.Local.SeventyB.SPARSEGPT_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-2-70b-hf_sparsegpt_unstructured_0.5"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.SPARSEGPT24_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-2-70b-hf_sparsegpt_2:4_0.5"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.SPARSEGPT_4BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-2-70b-hf_sparsegpt_unstructured_0.3"),
            "meta-llama/Llama-2-70b-hf"
        )
        self._register_model(
            Models.Llama3.Local.SeventyB.SPARSEGPT_2BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-2-70b-hf_sparsegpt_unstructured_0.1"),
            "meta-llama/Llama-2-70b-hf"
        )
        
    def _initialize_llama3_it_local_models(self):
        """Initialize Llama-3 Instruct local models"""
        base_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # QUANTO models
        self._register_model(
            Models.Llama3Instruct.Local.EightB.QUANTO_8BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-QUANTO-8"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.QUANTO_8BIT_CALIB,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-QUANTO-CALIB-8"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.QUANTO_MIXED,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-QUANTO-8-mixed"),
            base_path
        )

        # AWQ models
        self._register_model(
            Models.Llama3Instruct.Local.EightB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-AWQ-4bit"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.AWQ_4BIT_OASST,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-AWQ-4bit-OASST"),
            base_path
        )

        # BNB models
        self._register_model(
            Models.Llama3Instruct.Local.EightB.BNB_8BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-BNB-8"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.BNB_4BIT,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-BNB-4"),
            base_path
        )

        # HQQ models
        self._register_model(
            Models.Llama3Instruct.Local.EightB.HQQ_8BIT_UNIFORM,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-HQQ-8-uniform"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.HQQ_MIXED,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-HQQ-mixed"),
            base_path
        )
        self._register_model(
            Models.Llama3Instruct.Local.EightB.HQQ_LORA,
            os.path.join(self.model_save_path, "Meta-Llama-3-8B-Instruct-HQQ-LORA"),
            base_path
        )
        
    def _initialize_llama31_models(self):
        """Initialize Llama-3.1 models"""
        self._initialize_llama31_base_models()
        self._initialize_llama31_huggingface_models()
        self._initialize_llama31_local_models()

    def _initialize_llama31_base_models(self):
        """Initialize Llama-3.1 base models"""
        self._register_model(
            Models.Llama31.Base.LLAMA_31_8B,
            "meta-llama/Llama-3.1-8B"
        )
        self._register_model(
            Models.Llama31.Base.LLAMA_31_70B,
            "meta-llama/Llama-3.1-70B"
        )

    def _initialize_llama31_huggingface_models(self):
        """Initialize Llama-3.1 HuggingFace models"""
        self._initialize_llama31_huggingface_8b_models()

    def _initialize_llama31_huggingface_8b_models(self):
        """Initialize Llama-3.1 HuggingFace 8B models"""
        base_path = "meta-llama/Llama-3.1-8B"
        
        # AWQ models
        self._register_model(
            Models.Llama31.HuggingFace.EightB.AWQ_4BIT,
            "JonathanBenaharon/Llama-3.1-8B-AWQ-4bit",
            "JonathanBenaharon/Llama-3.1-8B-AWQ-4bit"
        )

        # HQQ models
        for model_config in [
            Models.Llama31.HuggingFace.EightB.HQQ_8BIT,
            Models.Llama31.HuggingFace.EightB.HQQ_4BIT,
            Models.Llama31.HuggingFace.EightB.HQQ_3BIT,
            Models.Llama31.HuggingFace.EightB.HQQ_2BIT,
            Models.Llama31.HuggingFace.EightB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama31.HuggingFace.EightB.QUANTO_8BIT,
            Models.Llama31.HuggingFace.EightB.QUANTO_4BIT,
            Models.Llama31.HuggingFace.EightB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)

        # BNB models
        for model_config in [
            Models.Llama31.HuggingFace.EightB.BNB_8BIT,
            Models.Llama31.HuggingFace.EightB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama31.HuggingFace.EightB.GPTQ_8BIT,
            Models.Llama31.HuggingFace.EightB.GPTQ_4BIT,
            Models.Llama31.HuggingFace.EightB.GPTQ_3BIT,
            Models.Llama31.HuggingFace.EightB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)
            
        # AQLM PV models
        self._register_model(
            Models.Llama31.HuggingFace.EightB.AQLM_PV_2BIT,
            "ISTA-DASLab/Meta-Llama-3.1-8B-AQLM-PV-2Bit-1x16-hf",
            "ISTA-DASLab/Meta-Llama-3.1-8B-AQLM-PV-2Bit-1x16-hf"
        )
        self._register_model(
            Models.Llama31.HuggingFace.EightB.AQLM_PV_1BIT,
            "ISTA-DASLab/Meta-Llama-3.1-8B-AQLM-PV-1Bit-1x16-hf",
            "ISTA-DASLab/Meta-Llama-3.1-8B-AQLM-PV-1Bit-1x16-hf"
        )

    def _initialize_llama31_local_models(self):
        """Initialize Llama-3.1 local models"""
        base_path = "meta-llama/Llama-3.1-8B"
        
        # AWQ models
        self._register_model(
            Models.Llama31.Local.EightB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Llama-31-8B-AWQ-4bit"),
            base_path
        )
        
    def _initialize_llama32_models(self):
        """Initialize Llama-3.2 models"""
        # Base Llama32 models
        self._initialize_llama32_base_models()
        self._initialize_llama32_huggingface_models()
        self._initialize_llama32_local_models()
        
        # Instruct Llama32 models
        self._initialize_llama32_it_base_models()
        self._initialize_llama32_it_huggingface_models()
        self._initialize_llama32_it_local_models()

    def _initialize_llama32_base_models(self):
        """Initialize Llama-3.2 base models"""
        self._register_model(
            Models.Llama32.Base.LLAMA_32_1B,
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Base.LLAMA_32_3B,
            "meta-llama/Llama-3.2-3B"
        )

    def _initialize_llama32_huggingface_models(self):
        """Initialize Llama-3.2 HuggingFace models"""
        self._initialize_llama32_huggingface_1b_models()
        self._initialize_llama32_huggingface_3b_models()

    def _initialize_llama32_huggingface_1b_models(self):
        """Initialize Llama-3.2 HuggingFace 1B models"""
        base_path = "meta-llama/Llama-3.2-1B"

        # AWQ models
        self._register_model(
            Models.Llama32.HuggingFace.OneB.AWQ_4BIT,
            "skshmjn/Llama-3.2-1B-awq-4-bit",
            "skshmjn/Llama-3.2-1B-awq-4-bit"
        )

        # BNB models
        for model_config in [
            Models.Llama32.HuggingFace.OneB.BNB_8BIT,
            Models.Llama32.HuggingFace.OneB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama32.HuggingFace.OneB.GPTQ_8BIT,
            Models.Llama32.HuggingFace.OneB.GPTQ_4BIT,
            Models.Llama32.HuggingFace.OneB.GPTQ_3BIT,
            Models.Llama32.HuggingFace.OneB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama32.HuggingFace.OneB.HQQ_8BIT,
            Models.Llama32.HuggingFace.OneB.HQQ_4BIT,
            Models.Llama32.HuggingFace.OneB.HQQ_3BIT,
            Models.Llama32.HuggingFace.OneB.HQQ_2BIT,
            Models.Llama32.HuggingFace.OneB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama32.HuggingFace.OneB.QUANTO_8BIT,
            Models.Llama32.HuggingFace.OneB.QUANTO_4BIT,
            Models.Llama32.HuggingFace.OneB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)
        
        # AQLM PV models
        self._register_model(
            Models.Llama32.HuggingFace.OneB.AQLM_PV_2BIT,
            "ISTA-DASLab/Llama-3.2-1B-AQLM-PV-2Bit-2x8",
            "ISTA-DASLab/Llama-3.2-1B-AQLM-PV-2Bit-2x8"
        )

    def _initialize_llama32_huggingface_3b_models(self):
        """Initialize Llama-3.2 HuggingFace 3B models"""
        base_path = "meta-llama/Llama-3.2-3B"

        # HQQ models
        for model_config in [
            Models.Llama32.HuggingFace.ThreeB.HQQ_8BIT,
            Models.Llama32.HuggingFace.ThreeB.HQQ_4BIT,
            Models.Llama32.HuggingFace.ThreeB.HQQ_3BIT,
            Models.Llama32.HuggingFace.ThreeB.HQQ_2BIT,
            Models.Llama32.HuggingFace.ThreeB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama32.HuggingFace.ThreeB.QUANTO_8BIT,
            Models.Llama32.HuggingFace.ThreeB.QUANTO_4BIT,
            Models.Llama32.HuggingFace.ThreeB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)

        # BNB models
        for model_config in [
            Models.Llama32.HuggingFace.ThreeB.BNB_8BIT,
            Models.Llama32.HuggingFace.ThreeB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama32.HuggingFace.ThreeB.GPTQ_8BIT,
            Models.Llama32.HuggingFace.ThreeB.GPTQ_4BIT,
            Models.Llama32.HuggingFace.ThreeB.GPTQ_3BIT,
            Models.Llama32.HuggingFace.ThreeB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)
            
        # AQLM PV models
        self._register_model(
            Models.Llama32.HuggingFace.ThreeB.AQLM_PV_2BIT,
            "ISTA-DASLab/Llama-3.2-3B-AQLM-PV-2Bit-2x8",
            "ISTA-DASLab/Llama-3.2-3B-AQLM-PV-2Bit-2x8"
        )

    def _initialize_llama32_local_models(self):
        """Initialize Llama-3.2 local models"""
        # 1B models
        self._register_model(
            Models.Llama32.Local.OneB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Llama-32-1B-AWQ-4bit"),
            "meta-llama/Llama-3.2-1B"
        )

        # 3B models
        self._register_model(
            Models.Llama32.Local.ThreeB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Llama-32-3B-AWQ-4bit"),
            "meta-llama/Llama-3.2-3B"
        )
        
        # Wanda models
        self._register_model(
            Models.Llama32.Local.OneB.WANDA_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-1B_wanda_unstructured_0.5"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.WANDA24_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-1B_wanda_2:4_0.5"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.WANDA_4BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-1B_wanda_unstructured_0.3"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.WANDA_2BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-1B_wanda_unstructured_0.1"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.WANDA_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-3B_wanda_unstructured_0.5"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.WANDA24_8BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-3B_wanda_2:4_0.5"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.WANDA_4BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-3B_wanda_unstructured_0.3"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.WANDA_2BIT,
            os.path.join(self.ceph_model_path, "wanda", "Llama-3.2-3B_wanda_unstructured_0.1"),
            "meta-llama/Llama-3.2-3B"
        )
        
        # SparseGPT models
        self._register_model(
            Models.Llama32.Local.OneB.SPARSEGPT_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-1B_sparsegpt_unstructured_0.5"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.SPARSEGPT24_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-1B_sparsegpt_2:4_0.5"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.SPARSEGPT_4BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-1B_sparsegpt_unstructured_0.3"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.OneB.SPARSEGPT_2BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-1B_sparsegpt_unstructured_0.1"),
            "meta-llama/Llama-3.2-1B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.SPARSEGPT_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-3B_sparsegpt_unstructured_0.5"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.SPARSEGPT24_8BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-3B_sparsegpt_2:4_0.5"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.SPARSEGPT_4BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-3B_sparsegpt_unstructured_0.3"),
            "meta-llama/Llama-3.2-3B"
        )
        self._register_model(
            Models.Llama32.Local.ThreeB.SPARSEGPT_2BIT,
            os.path.join(self.ceph_model_path, "sparsegpt", "Llama-3.2-3B_sparsegpt_unstructured_0.1"),
            "meta-llama/Llama-3.2-3B"
        )
        
    def _initialize_llama32_it_base_models(self):
        """Initialize Llama-3.2 Instruct base models"""
        self._register_model(
            Models.Llama32Instruct.Base.LLAMA_32_1B,
            "meta-llama/Llama-3.2-1B-Instruct"
        )
        self._register_model(
            Models.Llama32Instruct.Base.LLAMA_32_3B,
            "meta-llama/Llama-3.2-3B-Instruct"
        )
        
    def _initialize_llama32_it_huggingface_models(self):
        """Initialize Llama-3.2 Instruct HuggingFace models"""
        self._initialize_llama32_it_huggingface_1b_models()
        self._initialize_llama32_it_huggingface_3b_models()
        
    def _initialize_llama32_it_huggingface_1b_models(self):
        """Initialize Llama-3.2 Instruct HuggingFace 1B models"""
        base_path = "meta-llama/Llama-3.2-1B-Instruct"

        # AWQ models
        self._register_model(
            Models.Llama32Instruct.HuggingFace.OneB.AWQ_4BIT,
            "skshmjn/Llama-3.2-1B-Instruct-awq-4-bit",
            "skshmjn/Llama-3.2-1B-Instruct-awq-4-bit"
        )

        # BNB models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.OneB.BNB_8BIT,
            Models.Llama32Instruct.HuggingFace.OneB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.OneB.GPTQ_8BIT,
            Models.Llama32Instruct.HuggingFace.OneB.GPTQ_4BIT,
            Models.Llama32Instruct.HuggingFace.OneB.GPTQ_3BIT,
            Models.Llama32Instruct.HuggingFace.OneB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)

        # HQQ models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.OneB.HQQ_8BIT,
            Models.Llama32Instruct.HuggingFace.OneB.HQQ_4BIT,
            Models.Llama32Instruct.HuggingFace.OneB.HQQ_3BIT,
            Models.Llama32Instruct.HuggingFace.OneB.HQQ_2BIT,
            Models.Llama32Instruct.HuggingFace.OneB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.OneB.QUANTO_8BIT,
            Models.Llama32Instruct.HuggingFace.OneB.QUANTO_4BIT,
            Models.Llama32Instruct.HuggingFace.OneB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)
        
        # AQLM PV models
        self._register_model(
            Models.Llama32Instruct.HuggingFace.OneB.AQLM_PV_2BIT,
            "ISTA-DASLab/Llama-3.2-1B-Instruct-AQLM-PV-2Bit-2x8",
            "ISTA-DASLab/Llama-3.2-1B-Instruct-AQLM-PV-2Bit-2x8"
        )
        
        # QuIP models
        self._register_model(
            Models.Llama32Instruct.HuggingFace.OneB.QUIP_4BIT,
            "nm-testing/Llama-3.2-1B-Instruct-quip-w4a16",
            "nm-testing/Llama-3.2-1B-Instruct-quip-w4a16"
        )

    def _initialize_llama32_it_huggingface_3b_models(self):
        """Initialize Llama-3.2 Instruct HuggingFace 3B models"""
        base_path = "meta-llama/Llama-3.2-3B-Instruct"

        # HQQ models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.ThreeB.HQQ_8BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.HQQ_4BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.HQQ_3BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.HQQ_2BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.HQQ_1BIT
        ]:
            self._register_model(model_config, base_path)

        # QUANTO models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.ThreeB.QUANTO_8BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.QUANTO_4BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.QUANTO_2BIT
        ]:
            self._register_model(model_config, base_path)

        # BNB models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.ThreeB.BNB_8BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.BNB_4BIT
        ]:
            self._register_model(model_config, base_path)

        # GPTQ models
        for model_config in [
            Models.Llama32Instruct.HuggingFace.ThreeB.GPTQ_8BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.GPTQ_4BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.GPTQ_3BIT,
            Models.Llama32Instruct.HuggingFace.ThreeB.GPTQ_2BIT
        ]:
            self._register_model(model_config, base_path)
            
        # AQLM PV models
        self._register_model(
            Models.Llama32Instruct.HuggingFace.ThreeB.AQLM_PV_2BIT,
            "ISTA-DASLab/Llama-3.2-3B-Instruct-AQLM-PV-2Bit-2x8",
            "ISTA-DASLab/Llama-3.2-3B-Instruct-AQLM-PV-2Bit-2x8"
        )
        
    def _initialize_llama32_it_local_models(self):
        """Initialize Llama-3.2 Instruct local models"""
        # 1B models
        self._register_model(
            Models.Llama32Instruct.Local.OneB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Llama-32-1B-Instruct-AWQ-4bit"),
            "meta-llama/Llama-3.2-1B-Instruct"
        )

        # 3B models
        self._register_model(
            Models.Llama32Instruct.Local.ThreeB.AWQ_4BIT,
            os.path.join(self.model_save_path, "Llama-32-3B-Instruct-AWQ-4bit"),
            "meta-llama/Llama-3.2-3B-Instruct"
        )
        
    def _initialize_opt_base_models(self):
        """Initialize OPT models"""
        # Base Models
        self._register_model(
            Models.Opt.Base.OPT_125M,
            "facebook/opt-125m"
        )
        self._register_model(
            Models.Opt.Base.OPT_350M,
            "facebook/opt-350m"
        )
        self._register_model(
            Models.Opt.Base.OPT_1B,
            "facebook/opt-1.3b"
        )
        self._register_model(
            Models.Opt.Base.OPT_2B,
            "facebook/opt-2.7b"
        )
        self._register_model(
            Models.Opt.Base.OPT_6B,
            "facebook/opt-6.7b"
        )
        self._register_model(
            Models.Opt.Base.OPT_13B,
            "facebook/opt-13b"
        )
        self._register_model(
            Models.Opt.Base.OPT_30B,
            "facebook/opt-30b"
        )
        self._register_model(
            Models.Opt.Base.OPT_66B,
            "facebook/opt-66b"
        )
        
    def _initialize_opt_huggingface_models(self):
        """Initialize OPT HuggingFace models"""
        # AWQ models
        self._register_model(
            Models.Opt.HuggingFace.M125.AWQ_4BIT,
            "titanbot/opt-125m-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.M350.AWQ_4BIT,
            "titanbot/opt-350m-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B1_3.AWQ_4BIT,
            "titanbot/opt-1.3b-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B2_7.AWQ_4BIT,
            "titanbot/opt-2.7b-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B6_7.AWQ_4BIT,
            "titanbot/opt-6.7b-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B13.AWQ_4BIT,
            "titanbot/opt-13b-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B30.AWQ_4BIT,
            "titanbot/opt-30b-AWQ-4bit",
        )
        self._register_model(
            Models.Opt.HuggingFace.B66.AWQ_4BIT,
            "titanbot/opt-66b-AWQ-4bit",
        )
        # BNB models
        for model_config in [
            Models.Opt.HuggingFace.M125.BNB_8BIT,
            Models.Opt.HuggingFace.M125.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-125m")
        for model_config in [
            Models.Opt.HuggingFace.M350.BNB_8BIT,
            Models.Opt.HuggingFace.M350.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-350m")
        for model_config in [
            Models.Opt.HuggingFace.B1_3.BNB_8BIT,
            Models.Opt.HuggingFace.B1_3.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-1.3b")
        for model_config in [
            Models.Opt.HuggingFace.B2_7.BNB_8BIT,
            Models.Opt.HuggingFace.B2_7.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-2.7b")
        for model_config in [   
            Models.Opt.HuggingFace.B6_7.BNB_8BIT,
            Models.Opt.HuggingFace.B6_7.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-6.7b")
        for model_config in [
            Models.Opt.HuggingFace.B13.BNB_8BIT,
            Models.Opt.HuggingFace.B13.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-13b")
        for model_config in [
            Models.Opt.HuggingFace.B30.BNB_8BIT,
            Models.Opt.HuggingFace.B30.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-30b")
        for model_config in [
            Models.Opt.HuggingFace.B66.BNB_8BIT,
            Models.Opt.HuggingFace.B66.BNB_4BIT,
        ]:
            self._register_model(model_config, "facebook/opt-66b")
            
        # GPTQ models
        for model_config in [
            Models.Opt.HuggingFace.M125.GPTQ_8BIT,
            Models.Opt.HuggingFace.M125.GPTQ_4BIT,
            Models.Opt.HuggingFace.M125.GPTQ_3BIT,
            Models.Opt.HuggingFace.M125.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-125m")
        for model_config in [
            Models.Opt.HuggingFace.M350.GPTQ_8BIT,
            Models.Opt.HuggingFace.M350.GPTQ_4BIT,
            Models.Opt.HuggingFace.M350.GPTQ_3BIT,
            Models.Opt.HuggingFace.M350.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-350m")
        for model_config in [
            Models.Opt.HuggingFace.B1_3.GPTQ_8BIT,
            Models.Opt.HuggingFace.B1_3.GPTQ_4BIT,
            Models.Opt.HuggingFace.B1_3.GPTQ_3BIT,
            Models.Opt.HuggingFace.B1_3.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-1.3b")
        for model_config in [
            Models.Opt.HuggingFace.B2_7.GPTQ_8BIT,
            Models.Opt.HuggingFace.B2_7.GPTQ_4BIT,
            Models.Opt.HuggingFace.B2_7.GPTQ_3BIT,
            Models.Opt.HuggingFace.B2_7.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-2.7b")
        for model_config in [
            Models.Opt.HuggingFace.B6_7.GPTQ_8BIT,
            Models.Opt.HuggingFace.B6_7.GPTQ_4BIT,
            Models.Opt.HuggingFace.B6_7.GPTQ_3BIT,
            Models.Opt.HuggingFace.B6_7.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-6.7b")
        for model_config in [
            Models.Opt.HuggingFace.B13.GPTQ_8BIT,
            Models.Opt.HuggingFace.B13.GPTQ_4BIT,
            Models.Opt.HuggingFace.B13.GPTQ_3BIT,
            Models.Opt.HuggingFace.B13.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-13b")
        for model_config in [
            Models.Opt.HuggingFace.B30.GPTQ_8BIT,
            Models.Opt.HuggingFace.B30.GPTQ_4BIT,
            Models.Opt.HuggingFace.B30.GPTQ_3BIT,
            Models.Opt.HuggingFace.B30.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-30b")
        for model_config in [
            Models.Opt.HuggingFace.B66.GPTQ_8BIT,
            Models.Opt.HuggingFace.B66.GPTQ_4BIT,
            Models.Opt.HuggingFace.B66.GPTQ_3BIT,
            Models.Opt.HuggingFace.B66.GPTQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-66b")
        
        # HQQ models
        for model_config in [
            Models.Opt.HuggingFace.M125.HQQ_8BIT,
            Models.Opt.HuggingFace.M125.HQQ_4BIT,
            Models.Opt.HuggingFace.M125.HQQ_3BIT,
            Models.Opt.HuggingFace.M125.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-125m")
        for model_config in [
            Models.Opt.HuggingFace.M350.HQQ_8BIT,
            Models.Opt.HuggingFace.M350.HQQ_4BIT,
            Models.Opt.HuggingFace.M350.HQQ_3BIT,
            Models.Opt.HuggingFace.M350.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-350m")
        for model_config in [
            Models.Opt.HuggingFace.B1_3.HQQ_8BIT,
            Models.Opt.HuggingFace.B1_3.HQQ_4BIT,
            Models.Opt.HuggingFace.B1_3.HQQ_3BIT,
            Models.Opt.HuggingFace.B1_3.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-1.3b")
        for model_config in [
            Models.Opt.HuggingFace.B2_7.HQQ_8BIT,
            Models.Opt.HuggingFace.B2_7.HQQ_4BIT,
            Models.Opt.HuggingFace.B2_7.HQQ_3BIT,
            Models.Opt.HuggingFace.B2_7.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-2.7b")
        for model_config in [
            Models.Opt.HuggingFace.B6_7.HQQ_8BIT,
            Models.Opt.HuggingFace.B6_7.HQQ_4BIT,
            Models.Opt.HuggingFace.B6_7.HQQ_3BIT,
            Models.Opt.HuggingFace.B6_7.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-6.7b")
        for model_config in [
            Models.Opt.HuggingFace.B13.HQQ_8BIT,
            Models.Opt.HuggingFace.B13.HQQ_4BIT,
            Models.Opt.HuggingFace.B13.HQQ_3BIT,
            Models.Opt.HuggingFace.B13.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-13b")
        for model_config in [
            Models.Opt.HuggingFace.B30.HQQ_8BIT,
            Models.Opt.HuggingFace.B30.HQQ_4BIT,
            Models.Opt.HuggingFace.B30.HQQ_3BIT,
            Models.Opt.HuggingFace.B30.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-30b")
        for model_config in [
            Models.Opt.HuggingFace.B66.HQQ_8BIT,
            Models.Opt.HuggingFace.B66.HQQ_4BIT,
            Models.Opt.HuggingFace.B66.HQQ_3BIT,
            Models.Opt.HuggingFace.B66.HQQ_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-66b")

        # Quanto models
        for model_config in [
            Models.Opt.HuggingFace.M125.QUANTO_8BIT,
            Models.Opt.HuggingFace.M125.QUANTO_4BIT,
            Models.Opt.HuggingFace.M125.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-125m")
        for model_config in [
            Models.Opt.HuggingFace.M350.QUANTO_8BIT,
            Models.Opt.HuggingFace.M350.QUANTO_4BIT,
            Models.Opt.HuggingFace.M350.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-350m")
        for model_config in [
            Models.Opt.HuggingFace.B1_3.QUANTO_8BIT,
            Models.Opt.HuggingFace.B1_3.QUANTO_4BIT,
            Models.Opt.HuggingFace.B1_3.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-1.3b")
        for model_config in [
            Models.Opt.HuggingFace.B2_7.QUANTO_8BIT,
            Models.Opt.HuggingFace.B2_7.QUANTO_4BIT,
            Models.Opt.HuggingFace.B2_7.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-2.7b")
        for model_config in [
            Models.Opt.HuggingFace.B6_7.QUANTO_8BIT,
            Models.Opt.HuggingFace.B6_7.QUANTO_4BIT,
            Models.Opt.HuggingFace.B6_7.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-6.7b")
        for model_config in [
            Models.Opt.HuggingFace.B13.QUANTO_8BIT,
            Models.Opt.HuggingFace.B13.QUANTO_4BIT,
            Models.Opt.HuggingFace.B13.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-13b")
        for model_config in [
            Models.Opt.HuggingFace.B30.QUANTO_8BIT,
            Models.Opt.HuggingFace.B30.QUANTO_4BIT,
            Models.Opt.HuggingFace.B30.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-30b")
        for model_config in [
            Models.Opt.HuggingFace.B66.QUANTO_8BIT,
            Models.Opt.HuggingFace.B66.QUANTO_4BIT,
            Models.Opt.HuggingFace.B66.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "facebook/opt-66b")
            
    def _initialize_qwen25vl_base_models(self):
        """Initialize Qwen-2.5-VL base models"""
        self._register_model(
            Models.Qwen25VL.Base.QWEN_2_5_VL_3B,
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self._register_model(
            Models.Qwen25VL.Base.QWEN_2_5_VL_7B,
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self._register_model(
            Models.Qwen25VL.Base.QWEN_2_5_VL_32B,
            "Qwen/Qwen2.5-VL-32B-Instruct"
        )
        self._register_model(
            Models.Qwen25VL.Base.QWEN_2_5_VL_72B,
            "Qwen/Qwen2.5-VL-72B-Instruct"
        )
        
    def _initialize_qwen25vl_huggingface_models(self):
        """Initialize Qwen-2.5-VL HuggingFace models"""
        # AWQ models
        self._register_model(
            Models.Qwen25VL.HuggingFace.B3.AWQ_4BIT,
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
        )
        self._register_model(
            Models.Qwen25VL.HuggingFace.B7.AWQ_4BIT,
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        )
        self._register_model(
            Models.Qwen25VL.HuggingFace.B32.AWQ_4BIT,
            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
        )
        self._register_model(
            Models.Qwen25VL.HuggingFace.B72.AWQ_4BIT,
            "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
        )
        
        # BNB models
        for model_config in [
            Models.Qwen25VL.HuggingFace.B3.BNB_8BIT,
            Models.Qwen25VL.HuggingFace.B3.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-3B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B7.BNB_8BIT,
            Models.Qwen25VL.HuggingFace.B7.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-7B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B32.BNB_8BIT,
            Models.Qwen25VL.HuggingFace.B32.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-32B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B72.BNB_8BIT,
            Models.Qwen25VL.HuggingFace.B72.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-72B-Instruct")
            
        # GPTQ models
        for model_config in [
            Models.Qwen25VL.HuggingFace.B3.GPTQ_8BIT,
            Models.Qwen25VL.HuggingFace.B3.GPTQ_4BIT,
            Models.Qwen25VL.HuggingFace.B3.GPTQ_3BIT,
            Models.Qwen25VL.HuggingFace.B3.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-3B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B7.GPTQ_8BIT,
            Models.Qwen25VL.HuggingFace.B7.GPTQ_4BIT,
            Models.Qwen25VL.HuggingFace.B7.GPTQ_3BIT,
            Models.Qwen25VL.HuggingFace.B7.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-7B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B32.GPTQ_8BIT,
            Models.Qwen25VL.HuggingFace.B32.GPTQ_4BIT,
            Models.Qwen25VL.HuggingFace.B32.GPTQ_3BIT,
            Models.Qwen25VL.HuggingFace.B32.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-32B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B72.GPTQ_8BIT,
            Models.Qwen25VL.HuggingFace.B72.GPTQ_4BIT,
            Models.Qwen25VL.HuggingFace.B72.GPTQ_3BIT,
            Models.Qwen25VL.HuggingFace.B72.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-72B-Instruct")
            
        # HQQ models
        for model_config in [
            Models.Qwen25VL.HuggingFace.B3.HQQ_8BIT,
            Models.Qwen25VL.HuggingFace.B3.HQQ_4BIT,
            Models.Qwen25VL.HuggingFace.B3.HQQ_3BIT,
            Models.Qwen25VL.HuggingFace.B3.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-3B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B7.HQQ_8BIT,
            Models.Qwen25VL.HuggingFace.B7.HQQ_4BIT,
            Models.Qwen25VL.HuggingFace.B7.HQQ_3BIT,
            Models.Qwen25VL.HuggingFace.B7.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-7B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B32.HQQ_8BIT,
            Models.Qwen25VL.HuggingFace.B32.HQQ_4BIT,
            Models.Qwen25VL.HuggingFace.B32.HQQ_3BIT,
            Models.Qwen25VL.HuggingFace.B32.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-32B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B72.HQQ_8BIT,
            Models.Qwen25VL.HuggingFace.B72.HQQ_4BIT,
            Models.Qwen25VL.HuggingFace.B72.HQQ_3BIT,
            Models.Qwen25VL.HuggingFace.B72.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-72B-Instruct")
            
        # Quanto models
        for model_config in [
            Models.Qwen25VL.HuggingFace.B3.QUANTO_8BIT,
            Models.Qwen25VL.HuggingFace.B3.QUANTO_4BIT,
            Models.Qwen25VL.HuggingFace.B3.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-3B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B7.QUANTO_8BIT,
            Models.Qwen25VL.HuggingFace.B7.QUANTO_4BIT,
            Models.Qwen25VL.HuggingFace.B7.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-7B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B32.QUANTO_8BIT,
            Models.Qwen25VL.HuggingFace.B32.QUANTO_4BIT,
            Models.Qwen25VL.HuggingFace.B32.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-32B-Instruct")
        for model_config in [
            Models.Qwen25VL.HuggingFace.B72.QUANTO_8BIT,
            Models.Qwen25VL.HuggingFace.B72.QUANTO_4BIT,
            Models.Qwen25VL.HuggingFace.B72.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen2.5-VL-72B-Instruct")
            
    def _initialize_qwen3vl_base_models(self):
        """Initialize Qwen-3-VL base models"""
        self._register_model(
            Models.Qwen3VL.Base.QWEN_3_VL_2B,
            "Qwen/Qwen3-VL-2B-Instruct"
        )
        self._register_model(
            Models.Qwen3VL.Base.QWEN_3_VL_4B,
            "Qwen/Qwen3-VL-4B-Instruct"
        )
        self._register_model(
            Models.Qwen3VL.Base.QWEN_3_VL_8B,
            "Qwen/Qwen3-VL-8B-Instruct"
        )
        self._register_model(
            Models.Qwen3VL.Base.QWEN_3_VL_32B,
            "Qwen/Qwen3-VL-32B-Instruct"
        )
            
    def _initialize_qwen3vl_huggingface_models(self):
        """Initialize Qwen-3-VL HuggingFace models"""
        # BNB models
        for model_config in [
            Models.Qwen3VL.HuggingFace.B2.BNB_8BIT,
            Models.Qwen3VL.HuggingFace.B2.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-2B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B4.BNB_8BIT,
            Models.Qwen3VL.HuggingFace.B4.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-4B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B8.BNB_8BIT,
            Models.Qwen3VL.HuggingFace.B8.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-8B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B32.BNB_8BIT,
            Models.Qwen3VL.HuggingFace.B32.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-32B-Instruct")
            
        # GPTQ models
        for model_config in [
            Models.Qwen3VL.HuggingFace.B2.GPTQ_8BIT,
            Models.Qwen3VL.HuggingFace.B2.GPTQ_4BIT,
            Models.Qwen3VL.HuggingFace.B2.GPTQ_3BIT,
            Models.Qwen3VL.HuggingFace.B2.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-2B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B4.GPTQ_8BIT,
            Models.Qwen3VL.HuggingFace.B4.GPTQ_4BIT,
            Models.Qwen3VL.HuggingFace.B4.GPTQ_3BIT,
            Models.Qwen3VL.HuggingFace.B4.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-4B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B8.GPTQ_8BIT,
            Models.Qwen3VL.HuggingFace.B8.GPTQ_4BIT,
            Models.Qwen3VL.HuggingFace.B8.GPTQ_3BIT,
            Models.Qwen3VL.HuggingFace.B8.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-8B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B32.GPTQ_8BIT,
            Models.Qwen3VL.HuggingFace.B32.GPTQ_4BIT,
            Models.Qwen3VL.HuggingFace.B32.GPTQ_3BIT,
            Models.Qwen3VL.HuggingFace.B32.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-32B-Instruct")
            
        # HQQ models
        for model_config in [
            Models.Qwen3VL.HuggingFace.B2.HQQ_8BIT,
            Models.Qwen3VL.HuggingFace.B2.HQQ_4BIT,
            Models.Qwen3VL.HuggingFace.B2.HQQ_3BIT,
            Models.Qwen3VL.HuggingFace.B2.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-2B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B4.HQQ_8BIT,
            Models.Qwen3VL.HuggingFace.B4.HQQ_4BIT,
            Models.Qwen3VL.HuggingFace.B4.HQQ_3BIT,
            Models.Qwen3VL.HuggingFace.B4.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-4B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B8.HQQ_8BIT,
            Models.Qwen3VL.HuggingFace.B8.HQQ_4BIT,
            Models.Qwen3VL.HuggingFace.B8.HQQ_3BIT,
            Models.Qwen3VL.HuggingFace.B8.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-8B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B32.HQQ_8BIT,
            Models.Qwen3VL.HuggingFace.B32.HQQ_4BIT,
            Models.Qwen3VL.HuggingFace.B32.HQQ_3BIT,
            Models.Qwen3VL.HuggingFace.B32.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-32B-Instruct")
            
        # Quanto models
        for model_config in [
            Models.Qwen3VL.HuggingFace.B2.QUANTO_8BIT,
            Models.Qwen3VL.HuggingFace.B2.QUANTO_4BIT,
            Models.Qwen3VL.HuggingFace.B2.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-2B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B4.QUANTO_8BIT,
            Models.Qwen3VL.HuggingFace.B4.QUANTO_4BIT,
            Models.Qwen3VL.HuggingFace.B4.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-4B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B8.QUANTO_8BIT,
            Models.Qwen3VL.HuggingFace.B8.QUANTO_4BIT,
            Models.Qwen3VL.HuggingFace.B8.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-8B-Instruct")
        for model_config in [
            Models.Qwen3VL.HuggingFace.B32.QUANTO_8BIT,
            Models.Qwen3VL.HuggingFace.B32.QUANTO_4BIT,
            Models.Qwen3VL.HuggingFace.B32.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-VL-32B-Instruct")
            
    def _initialize_qwen3_base_models(self):
        """Initialize Qwen-3 base models"""
        self._register_model(
            Models.Qwen3.Base.QWEN_3_1B,
            "Qwen/Qwen3-1.7B"
        )
        self._register_model(
            Models.Qwen3.Base.QWEN_3_4B,
            "Qwen/Qwen3-4B"
        )
        self._register_model(
            Models.Qwen3.Base.QWEN_3_8B,
            "Qwen/Qwen3-8B"
        )
        self._register_model(
            Models.Qwen3.Base.QWEN_3_14B,
            "Qwen/Qwen3-14B"
        )
        self._register_model(
            Models.Qwen3.Base.QWEN_3_32B,
            "Qwen/Qwen3-32B"
        )
        
    def _initialize_qwen3_huggingface_models(self):
        """Initialize Qwen-3 HuggingFace models"""
        # AWQ models
        self._register_model(
            Models.Qwen3.HuggingFace.B1_7.AWQ_4BIT,
            "flin775/Qwen3-1.7B-AWQ-Group32",
            "flin775/Qwen3-1.7B-AWQ-Group32"
        )
        self._register_model(
            Models.Qwen3.HuggingFace.B4.AWQ_4BIT,
            "Qwen/Qwen3-4B-AWQ",
            "Qwen/Qwen3-4B-AWQ"
        )
        self._register_model(
            Models.Qwen3.HuggingFace.B8.AWQ_4BIT,
            "Qwen/Qwen3-8B-AWQ",
            "Qwen/Qwen3-8B-AWQ"
        )
        self._register_model(
            Models.Qwen3.HuggingFace.B14.AWQ_4BIT,
            "Qwen/Qwen3-14B-AWQ",
            "Qwen/Qwen3-14B-AWQ"
        )
        self._register_model(
            Models.Qwen3.HuggingFace.B32.AWQ_4BIT,
            "Qwen/Qwen3-32B-AWQ",
            "Qwen/Qwen3-32B-AWQ"
        )
        
        # BNB models
        for model_config in [
            Models.Qwen3.HuggingFace.B1_7.BNB_8BIT,
            Models.Qwen3.HuggingFace.B1_7.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-1.7B")
        for model_config in [
            Models.Qwen3.HuggingFace.B4.BNB_8BIT,
            Models.Qwen3.HuggingFace.B4.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-4B")
        for model_config in [
            Models.Qwen3.HuggingFace.B8.BNB_8BIT,
            Models.Qwen3.HuggingFace.B8.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-8B")
        for model_config in [
            Models.Qwen3.HuggingFace.B14.BNB_8BIT,
            Models.Qwen3.HuggingFace.B14.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-14B")
        for model_config in [
            Models.Qwen3.HuggingFace.B32.BNB_8BIT,
            Models.Qwen3.HuggingFace.B32.BNB_4BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-32B")
        
        # GPTQ models
        for model_config in [
            Models.Qwen3.HuggingFace.B1_7.GPTQ_8BIT,
            Models.Qwen3.HuggingFace.B1_7.GPTQ_4BIT,
            Models.Qwen3.HuggingFace.B1_7.GPTQ_3BIT,
            Models.Qwen3.HuggingFace.B1_7.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-1.7B")
        for model_config in [
            Models.Qwen3.HuggingFace.B4.GPTQ_8BIT,
            Models.Qwen3.HuggingFace.B4.GPTQ_4BIT,
            Models.Qwen3.HuggingFace.B4.GPTQ_3BIT,
            Models.Qwen3.HuggingFace.B4.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-4B")
        for model_config in [
            Models.Qwen3.HuggingFace.B8.GPTQ_8BIT,
            Models.Qwen3.HuggingFace.B8.GPTQ_4BIT,
            Models.Qwen3.HuggingFace.B8.GPTQ_3BIT,
            Models.Qwen3.HuggingFace.B8.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-8B")
        for model_config in [
            Models.Qwen3.HuggingFace.B14.GPTQ_8BIT,
            Models.Qwen3.HuggingFace.B14.GPTQ_4BIT,
            Models.Qwen3.HuggingFace.B14.GPTQ_3BIT,
            Models.Qwen3.HuggingFace.B14.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-14B")
        for model_config in [
            Models.Qwen3.HuggingFace.B32.GPTQ_8BIT,
            Models.Qwen3.HuggingFace.B32.GPTQ_4BIT,
            Models.Qwen3.HuggingFace.B32.GPTQ_3BIT,
            Models.Qwen3.HuggingFace.B32.GPTQ_2BIT
        ]:
            self._register_model(model_config, "Qwen/Qwen3-32B")
            
        # HQQ models
        for model_config in [
            Models.Qwen3.HuggingFace.B1_7.HQQ_8BIT,
            Models.Qwen3.HuggingFace.B1_7.HQQ_4BIT,
            Models.Qwen3.HuggingFace.B1_7.HQQ_3BIT,
            Models.Qwen3.HuggingFace.B1_7.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-1.7B")
        for model_config in [
            Models.Qwen3.HuggingFace.B4.HQQ_8BIT,
            Models.Qwen3.HuggingFace.B4.HQQ_4BIT,
            Models.Qwen3.HuggingFace.B4.HQQ_3BIT,
            Models.Qwen3.HuggingFace.B4.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-4B")
        for model_config in [
            Models.Qwen3.HuggingFace.B8.HQQ_8BIT,
            Models.Qwen3.HuggingFace.B8.HQQ_4BIT,
            Models.Qwen3.HuggingFace.B8.HQQ_3BIT,
            Models.Qwen3.HuggingFace.B8.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-8B")
        for model_config in [
            Models.Qwen3.HuggingFace.B14.HQQ_8BIT,
            Models.Qwen3.HuggingFace.B14.HQQ_4BIT,
            Models.Qwen3.HuggingFace.B14.HQQ_3BIT,
            Models.Qwen3.HuggingFace.B14.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-14B")
        for model_config in [
            Models.Qwen3.HuggingFace.B32.HQQ_8BIT,
            Models.Qwen3.HuggingFace.B32.HQQ_4BIT,
            Models.Qwen3.HuggingFace.B32.HQQ_3BIT,
            Models.Qwen3.HuggingFace.B32.HQQ_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-32B")
            
        # Quanto models
        for model_config in [
            Models.Qwen3.HuggingFace.B1_7.QUANTO_8BIT,
            Models.Qwen3.HuggingFace.B1_7.QUANTO_4BIT,
            Models.Qwen3.HuggingFace.B1_7.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-1.7B")
        for model_config in [
            Models.Qwen3.HuggingFace.B4.QUANTO_8BIT,
            Models.Qwen3.HuggingFace.B4.QUANTO_4BIT,
            Models.Qwen3.HuggingFace.B4.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-4B")
        for model_config in [
            Models.Qwen3.HuggingFace.B8.QUANTO_8BIT,
            Models.Qwen3.HuggingFace.B8.QUANTO_4BIT,
            Models.Qwen3.HuggingFace.B8.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-8B")
        for model_config in [
            Models.Qwen3.HuggingFace.B14.QUANTO_8BIT,
            Models.Qwen3.HuggingFace.B14.QUANTO_4BIT,
            Models.Qwen3.HuggingFace.B14.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-14B")
        for model_config in [
            Models.Qwen3.HuggingFace.B32.QUANTO_8BIT,
            Models.Qwen3.HuggingFace.B32.QUANTO_4BIT,
            Models.Qwen3.HuggingFace.B32.QUANTO_2BIT,
        ]:
            self._register_model(model_config, "Qwen/Qwen3-32B")

    def _register_model(self, identifier: ModelIdentifier, model_path: str, tokenizer_path: Optional[str] = None):
        """Register a model with its paths"""
        if identifier.is_local:
            model_path = os.path.join(self.model_save_path, model_path)
        
        tokenizer_path = tokenizer_path or model_path
        self._model_paths[identifier] = ModelPaths(model_path, tokenizer_path)

    def get_model_paths(self, identifier: ModelIdentifier) -> Optional[ModelPaths]:
        """Get model paths for a given identifier"""
        return self._model_paths.get(identifier)

    def get_all_models(self) -> Dict[str, ModelPaths]:
        """Get all registered models with their paths"""
        return {str(identifier): paths for identifier, paths in self._model_paths.items()}

    def get_model_bit_precision(self, identifier: ModelIdentifier) -> int:
        """Get bit precision for a given model"""
        if identifier.bits:
            return identifier.bits.value
        return BitPrecision.FP16.value  # Default to FP16 for base models