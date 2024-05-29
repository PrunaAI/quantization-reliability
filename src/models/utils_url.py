import torch


def list_models(task):
    if task == "image_object_detection":
        return {}
    elif task == "image_instance_segmentation":
        return {}
    elif task == "image_classification":
        return {}
    else:
        raise NotImplementedError


get_url = {}


def get_model(name, weights):
    return torch.utils.model_zoo.load_url(get_url[name][weights])
