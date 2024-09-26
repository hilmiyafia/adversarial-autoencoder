
import torch
import torchvision


class VGGLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        self.vgg = torchvision.models.vgg16(weights=weights).features[:9]
        self.normalize = torchvision.transforms.Normalize(
            [0.48235, 0.45882, 0.40784],
            [0.00392156862745098, 0.00392156862745098, 0.00392156862745098])

    def forward(self, input):
        return self.vgg(self.normalize(input))

