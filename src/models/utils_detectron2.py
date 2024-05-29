import torch

try:
    from detectron2 import model_zoo
except ModuleNotFoundError:
    pass


def list_models(task):
    # Models loaded from Detectron2 repository: https://github.com/facebookresearch/detectron2/tree/main/configs
    if task == "image_object_detection":
        return {
            "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml",
            "COCO-Detection/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
            "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
            "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
            "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
            "COCO-Detection/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-Detection/fcos_R_50_FPN_1x.py",
            "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
            "COCO-Detection/retinanet_R_50_FPN_1x.py",
            "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
            "COCO-Detection/rpn_R_50_C4_1x.yaml",
            "COCO-Detection/rpn_R_50_FPN_1x.yaml",
            "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml",
            "PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml",
        }
    elif task == "image_instance_segmentation":
        return {
            "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.py",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py",
            "COCO-InstanceSegmentation/mask_rcnn_regnety_4gf_dds_fpn_1x.py",
            "Cityscapes/mask_rcnn_R_50_FPN.yaml",
            "LVISv1-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
            "LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
            "LVISv1-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml",
        }
    elif task == "image_keypoint_detection":
        return {
            "COCO-Keypoints/Base-Keypoint-RCNN-FPN.yaml",
            "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.py",
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml",
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
            "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml",
        }
    else:
        raise NotImplementedError


def get_model(name, pretrained=True):
    model = Detectron2Wrapper(model_zoo.get(name, trained=pretrained).eval())
    return model


class Detectron2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        batch_size, n_channels, height, width = x.shape

        inputs = []
        for b in range(batch_size):
            inputs.append({"image": x[b, :, :, :].squeeze() * 255, "height": height, "width": width})

        ys = self.model(inputs)

        y_list = []
        for y in ys:
            y_list.append(
                [
                    y["instances"].pred_boxes.tensor,
                    y["instances"].scores,
                    y["instances"].pred_classes,
                    y["instances"].pred_masks.long(),
                ]
            )
        return y_list
        # ys = self.model(inputs)[0]["instances"]
        #
        # return ys.pred_boxes.tensor, ys.scores, ys.pred_classes, ys.pred_masks.long()
