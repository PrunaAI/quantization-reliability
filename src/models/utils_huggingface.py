import torch
import json
from huggingface_hub import HfApi
from huggingface_hub import ModelFilter
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, WhisperForConditionalGeneration
from transformers import pipeline
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
)


def list_models(task):
    # TODO: Extend HF set_models with call to their API
    # TODO: Note that the filters might miss some interesting models because HF tags might be missing for some models.
    if task == "image-classification":
        set_models = set([])
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="image-classification",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "image-segmentation":
        set_models = set([])
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="image-segmentation",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "image-object-detection":
        set_models = set([])
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="object-detection",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "text-to-image-generation":
        set_models = {
            "CompVis/stable-diffusion-v1-4",
            "hakurei/waifu-diffusion",
            "hakurei/waifu-diffusion-v1-4",
            "SG161222/Realistic_Vision_V1.4",
            "runwayml/stable-diffusion-v1-5",
            "nitrosocke/Arcane-Diffusion",
            "wavymulder/Analog-Diffusion",
            "nitrosocke/redshift-diffusion",
            "prompthero/openjourney-v4",
            "darkstorm2150/Protogen_v5.8_Official_Release",
            "yehiaserag/anime-pencil-diffusion",
            "emilianJR/epiCRealism",
            "dreamlike-art/dreamlike-anime-1.0",
            "dreamlike-art/dreamlike-diffusion-1.0",
            "dreamlike-art/dreamlike-photoreal-1.0",
            "dreamlike-art/dreamlike-photoreal-2.0",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-0.9",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "SG161222/RealVisXL_V1.0",
            "SG161222/RealVisXL_V2.0",
            "SG161222/RealVisXL_V3.0",
            "SG161222/RealVisXL_V3.0_Turbo",
            "Linaqruf/animagine-xl",
            "nerijs/pixel-art-xl",
            "prompthero/openjourney",
            "IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B",
            "stabilityai/sdxl-turbo",
            "segmind/SSD-1B",
            "runwayml/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "segmind/Segmind-Vega",
            "SimianLuo/LCM_Dreamshaper_v7",
            "Lykon/dreamshaper-xl-v2-turbo",
        }
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="text-to-image",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "text-to-video-generation":
        set_models = {
            "damo-vilab/text-to-video-ms-1.7b",
            "cerspense/zeroscope_v2_576w",
        }
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="text-to-video",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "image-to-video-generation":
        set_models = {"stabilityai/stable-video-diffusion-img2vid-xt"}
        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                task="image-to-video",
                library="pytorch",
            )
        )
        for model in models:
            set_models.add(model.modelId)
        return set_models
    elif task == "text-to-text-generation":
        # Add Causal LM Huggingface models
        set_models = {
            "cerebras/btlm-3b-8k-base",
            "tiiuae/falcon-7b",
            "microsoft/phi-1_5",
            "mistralai/Mistral-7B-v0.1",
            "kittn/mistral-7B-v0.1-hf",
            "zjunlp/MolGen-large",
            "zjunlp/llama-molinst-protein-7b",
            "HuggingFaceH4/zephyr-7b-beta",
            "HuggingFaceH4/zephyr-7b-alpha",
            "gpt2",
            "gpt2-medium",
            "gpt2-xl",
            "gpt2-large",
            "mattshumer/Hermes-2-Pro-11B",
            "facebook/opt-13b",
            "gradientai/Llama-3-8B-Instruct-262k",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
            "cognitivecomputations/dolphin-2.9-llama3-8b",
            "cognitivecomputations/dolphin-2.9-llama3-8b-256k",
            "shenzhi-wang/Llama3-8B-Chinese-Chat",
            "aaditya/Llama3-OpenBioLLM-8B",
            "McGill-NLP/Llama-3-8B-Web",
            "chargoddard/llama3-42b-v0",
            "Orenguteng/Llama-3-8B-Lexi-Uncensored",
            "beomi/Llama-3-Open-Ko-8B",
            "openlynn/Llama-3-Soliloquy-8B",
            "UnicomLLM/Unichat-llama3-Chinese-8B",
            "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
            "openlynn/Llama-3-Soliloquy-8B-v1-24k",
            "Undi95/Llama-3-Unholy-8B",
            "bigcode/starcoder2-15b-instruct-v0.1",
            "google/gemma-7b",
            "google/gemma-7b-it",
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/codegemma-7b-it",
            "google/codegemma-7b",
            "google/codegemma-2b",
            "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "NeverSleep/Llama-3-Lumimaid-8B-v0.1",
            "Ppoyaa/Luna-8B-Instruct-262k",
            "Weyaxi/Einstein-v6.1-Llama3-8B",
            "nvidia/Llama3-ChatQA-1.5-8B",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "ytu-ce-cosmos/turkish-gpt2-large",
            "saltlux/Ko-Llama3-Luxia-8B",
            "ytu-ce-cosmos/turkish-gpt2-medium",
            "ytu-ce-cosmos/turkish-gpt2",
        }

        api = HfApi()
        # for causal_lm_name in list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.keys()):
        #     models = api.list_models(
        #         search=causal_lm_name,
        #         filter=ModelFilter(
        #             task="text-generation",
        #             library="pytorch",
        #         ),
        #     )
        #     for model in models:
        #         set_models.add(model.modelId)
        models = api.list_models(
            filter=ModelFilter(
                task="text-generation",
                library="pytorch",
            ),
        )
        for model in models:
            set_models.add(model.modelId)

        return set_models
    elif task == "text-to-text-translation":
        return {
            "facebook/bart-large-cnn",
            "zjunlp/MolGen-large",
        }
    elif task == "audio-to-text-transcription":
        return {"openai/whisper-tiny"}
    else:
        raise NotImplementedError


def get_causallm_model(name, weights, cache_dir, task, token=None):
    """
    Fetches a pre-trained language model from Hugging Face and wraps it for usage.

    Args:
    - model_name (str): The name or path of the pre-trained model on Hugging Face's model hub.

    Returns:
    - HuggingFaceLLMWrapper: A CUDA-accelerated instance of the model wrapped for further processing.
    """
    if task == "text-to-text-generation":
        # Wrap the pretrained model in the custom wrapper and load it onto GPU
        return AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype="auto",
            cache_dir=cache_dir,
            token=token,
        )
    else:
        raise NotImplementedError


def get_seq2seqlm_model(model_name, cache_dir):
    """
    Fetches a pre-trained translation language model from Hugging Face and wraps it for usage.

    Args:
    - model_name (str): The name or path of the pre-trained model on Hugging Face's model hub.

    Returns:
    - HuggingFaceLLMWrapper: A CUDA-accelerated instance of the model wrapped for further processing.
    """
    # Wrap the pretrained model in the custom wrapper and load it onto GPU
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype="auto", cache_dir=cache_dir
    )


def get_whisper_model(model_name, cache_dir):
    """
    Fetches a pre-trained language model from Hugging Face and wraps it for usage.

    Args:
    - model_name (str): The name or path of the pre-trained model on Hugging Face's model hub.

    Returns:
    - HuggingFaceLLMWrapper: A CUDA-accelerated instance of the model wrapped for further processing.
    """
    # Wrap the pretrained model in the custom wrapper and load it onto GPU

    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        torch_dtype=torch.float16,
        device="cuda",
    )


def get_diffusers_model(model_name, weight_name, task="txt2img", cache_dir=None):
    """
    Load a specific diffusion model pipeline.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    task : str, optional
        The name of the weights to use. Options include:
        - "txt2img": For text-to-image pipelines
        - "txt2imgxl": For XL version of text-to-image pipelines
        - "img2img": For image-to-image pipelines
        - "inpaint": For inpainting pipelines
        - "controlnet": For pipelines that use ControlNet models
        Default is 'txt2img'.
    cache_dir : str, optional
        Path to the cache directory. Default is None.

    Returns
    -------
    object
        Initialized pipeline object.

    Examples
    --------
    >>> get_model("runwayml/stable-diffusion-v1-5")
    <pipeline_object>
    """
    if "task" in smash_config_mapping[model_name]:
        task = smash_config_mapping[model_name]["task"]
    image_height = smash_config_mapping[model_name]["image_height"]
    image_width = smash_config_mapping[model_name]["image_width"]
    version = smash_config_mapping[model_name]["version"]

    if (
        task == "text_image_generation" or task == "text_video_generation" or task == "image_video_generation"
    ) and "xl" not in version:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        if weight_name is not None:
            pipe.unet.load_attn_procs(weight_name)
    elif task == "text_image_generation" and "xl" in version:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=cache_dir,
        )
        if weight_name is not None:
            pipe.unet.load_attn_procs(weight_name)
    elif task == "image_image_generation":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
    elif task == "image_image_inpainting":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
    # For this to be called we need to pass the controlnet model in weight_name and task="controlnet"
    elif task == "image_image_control":
        controlnet = ControlNetModel.from_pretrained(weight_name, torch_dtype=torch.float16, cache_dir=cache_dir)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, cache_dir=cache_dir
        )
    elif task == "image_image_control" and "xl" in version:
        controlnet = ControlNetModel.from_pretrained(weight_name, torch_dtype=torch.float16, cache_dir=cache_dir)

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet, torch_dtype=torch.float16, cache_dir=cache_dir
        )

    pipe = pipe

    pipe.model_name = model_name
    pipe.task = task
    pipe.weight_name = weight_name
    pipe.image_height = image_height
    pipe.image_width = image_width
    pipe.version = version
    return pipe
