import torch

import torchvision
from torchvision.models import list_models as torchvision_list_models
from torchvision.models import get_model as torchvision_get_model
from src.models.utils_torchvision import TVSegmentationWrapper, TVDetectionWrapper, TVVideoWrapper

# from src.models.utils_ultralitics import list_models as ultralitics_list_models
# from src.models.utils_ultralitics import UltraliticsDetectionWrapper, UltraliticsSegmentationWrapper

from src.models.utils_huggingface import list_models as huggingface_list_models
from src.models.utils_huggingface import get_causallm_model as huggingface_get_causallm_model
from src.models.utils_huggingface import get_whisper_model as huggingface_get_whisper_model
from src.models.utils_huggingface import get_seq2seqlm_model as huggingface_get_translation_model
from src.models.utils_huggingface import get_diffusers_model as huggingface_get_diffusers_model

from src.models.utils_civitai import list_models as civitai_list_models
from src.models.utils_civitai import get_model as civitai_get_model

from src.models.utils_url import list_models as url_list_models
from src.models.utils_url import get_model as url_get_model

from src.models.utils_detectron2 import list_models as detectron2_list_models
from src.models.utils_detectron2 import get_model as detectron2_get_model

from timm import list_models as timm_list_models
from timm import create_model as timm_get_model


def get_model(model_name, weight_name, cache_dir, token=None):
    # Torchvision
    if model_name in torchvision_list_models(module=torchvision.models):
        return torchvision_get_model(name=model_name, weights=weight_name)
    elif model_name in torchvision_list_models(module=torchvision.models.segmentation):
        return TVSegmentationWrapper(torchvision_get_model(name=model_name, weights=weight_name))
    elif model_name in torchvision_list_models(module=torchvision.models.detection):
        return TVDetectionWrapper(torchvision_get_model(name=model_name, weights=weight_name))
    elif model_name in torchvision_list_models(module=torchvision.models.video):
        return TVVideoWrapper(torchvision_get_model(name=model_name, weights=weight_name))
    # # Ultralitics
    # elif model_name in ultralitics_list_models(task="image_classification"):
    #     return YOLO(f"{model_name}.pt").model
    # elif model_name in ultralitics_list_models(task="image_object_detection"):
    #     return UltraliticsDetectionWrapper(YOLO(f"{model_name}.pt").model)
    # elif model_name in ultralitics_list_models(task="image_instance_segmentation"):
    #     return UltraliticsSegmentationWrapper(YOLO(f"{model_name}.pt").model)
    # elif model_name in ultralitics_list_models(task="image_keypoint_detection"):
    #     return UltraliticsSegmentationWrapper(YOLO(f"{model_name}.pt").model)
    # Detectron2
    elif model_name in detectron2_list_models(task="image_object_detection"):
        return detectron2_get_model(model_name, pretrained=True)
    elif model_name in detectron2_list_models(task="image_instance_segmentation"):
        return detectron2_get_model(model_name, pretrained=True)
    elif model_name in detectron2_list_models(task="image_keypoint_detection"):
        return detectron2_get_model(model_name, pretrained=True)
    # TIMM
    elif model_name in timm_list_models(pretrained=True):
        return timm_get_model(model_name, pretrained=True)
    # Huggingface
    elif model_name in huggingface_list_models(task="text-to-text-generation"):
        return huggingface_get_causallm_model(
            name=model_name, weights=weight_name, cache_dir=cache_dir, task="text-to-text-generation", token=token
        )
    elif model_name in huggingface_list_models(task="text-to-text-translation"):
        return huggingface_get_translation_model(model_name=model_name, cache_dir=cache_dir)
    elif model_name in huggingface_list_models(task="audio-to-text-transcription"):
        return huggingface_get_whisper_model(model_name=model_name, cache_dir=cache_dir)
    elif model_name in huggingface_list_models(task="text-to-image-generation"):
        return huggingface_get_diffusers_model(model_name=model_name, weight_name=weight_name, cache_dir=cache_dir)
    elif model_name in huggingface_list_models(task="text-to-video-generation"):
        return huggingface_get_diffusers_model(model_name=model_name, weight_name=weight_name, cache_dir=cache_dir)
    elif model_name in huggingface_list_models(task="image-to-video-generation"):
        return huggingface_get_diffusers_model(model_name=model_name, weight_name=weight_name, cache_dir=cache_dir)
    # Civitai
    elif model_name in civitai_list_models(task="text-to-image-generation"):
        return civitai_get_model(model_name, weight_name, cache_dir=cache_dir)
    # URL
    elif model_name in url_list_models(task="image_classification"):
        return url_get_model(name=model_name, weights=weight_name)
    else:
        raise NotImplementedError
