import torch


def list_models(task):
    if task == "image_object_detection":
        return {
            "yolov3",
            "yolov3u",
            "yolov5nu",
            "yolov5su",
            "yolov5mu",
            "yolov5lu",
            "yolov5xu",
            "yolov8n",
            "yolov8s",
            "yolov8l",
            "yolov8m",
            "yolov8x",
        }
    elif task == "image_instance_segmentation":
        return {
            "yolov8n-seg",
            "yolov8s-seg",
            "yolov8l-seg",
            "yolov8m-seg",
            "yolov8x-seg",
        }
    elif task == "image_keypoint_detection":
        return {
            "yolov8n-pose",
            "yolov8s-pose",
            "yolov8l-pose",
            "yolov8m-pose",
            "yolov8x-pose",
        }
    elif task == "image_classification":
        return {
            "yolov8n-cls",
            "yolov8s-cls",
            "yolov8l-cls",
            "yolov8m-cls",
            "yolov8x-cls",
        }
    else:
        raise NotImplementedError


class UltraliticsDetectionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        ys = self.model(x)

        return ys[0], ys[1][0], ys[1][1], ys[1][2]


class UltraliticsSegmentationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        ys = self.model(x)

        return ys[0], ys[1][0][0], ys[1][0][1], ys[1][0][2], ys[1][1]
