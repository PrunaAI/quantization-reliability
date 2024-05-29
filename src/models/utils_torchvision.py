import torch


class TVSegmentationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        y = self.model(x)
        return y["out"]


class TVDetectionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        ys = self.model(x)
        y_list = []
        for y in ys:
            y_list.append(
                [
                    y["boxes"],
                    y["scores"],
                    y["labels"],
                    y["masks"],
                ]
            )
        return y_list


class TVVideoWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
        return
