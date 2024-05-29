import os
import requests
import subprocess
import torch

from src.algorithms.smash_config_mapping import smash_config_mapping
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
    if task == "text-to-image-generation":
        set_models = {
            "chinese-landscape-art",
            "picxreal",
            "pony-diffusion-v6-xl",
            "zavychromaxl",
            "bluepencil-xl",
            "newrealityxl-all-in-one-photographic",
            "copax-timelessxl-sdxl10",
            "realism-engine-sdxl",
            "sdxl-yamers-realistic-5",
            "leosams-helloworld-sdxl-base-model",
            "realities-edge-xl-sdxl-turbo",
            "devlishphotorealism-sdxl",
            "dreamshaper",
            "absolutereality",
            "indigo-furry-mix",
            "sdxlnijispecial-edition",
            "starlight-xl-animated",
            "wesumix-real-fantasy-5",
            "juggernaut",
            "juggernaut-xl",
            "leosams-helloworld-xl",
            "leosams-moonfilm-fp16cleaned",
            "ultraspice",
            "paradox-2-sd-xl-10",
        }
        return set_models
    else:
        raise NotImplementedError


def get_model(model_name, weight_name, cache_dir=None):
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
    >>> get_model("pony-diffusion-v6-xl")
    <pipeline_object>
    """
    if "task" in smash_config_mapping[model_name]:
        task = smash_config_mapping[model_name]["task"]
    image_height = smash_config_mapping[model_name]["image_height"]
    image_width = smash_config_mapping[model_name]["image_width"]
    version = smash_config_mapping[model_name]["version"]
    civitai_url = smash_config_mapping[model_name]["model_url"]

    # Download model
    cached_model_path = os.path.join(cache_dir, model_name + ".safetensors")
    response = requests.get(civitai_url)
    if response.status_code == 200:
        with open(cached_model_path, "wb") as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download model from {civitai_url}")

    # Load model
    if task == "text_image_generation" and "xl" not in version:
        pipe = StableDiffusionPipeline.from_single_file(cached_model_path, torch_dtype=torch.float16)
        if weight_name is not None:
            pipe.unet.load_attn_procs(weight_name)
    elif task == "text_image_generation" and "xl" in version:
        pipe = StableDiffusionXLPipeline.from_single_file(
            cached_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        if weight_name is not None:
            pipe.unet.load_attn_procs(weight_name)
    else:
        raise NotImplementedError

    pipe = pipe

    pipe.model_name = model_name
    pipe.task = task
    pipe.weight_name = weight_name
    pipe.image_height = image_height
    pipe.image_width = image_width
    pipe.version = version
    return pipe
