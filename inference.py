
import cv2
import numpy
import torch
import torchvision
from train import AdversarialAutoencoder


ROWS = 4
COLS = 4


if __name__ == "__main__":
    model = AdversarialAutoencoder.load_from_checkpoint("model.ckpt")
    model.eval()

    with torch.no_grad():
        random = torch.randn(ROWS*COLS, 256)
        random = random.as_type(model.autoencoder.decoder[-1].layers[0].weight)
        random = model.autoencoder.decoder(random)
        random = torch.concat([torch.concat([random[i+j*ROWS] for i in range(ROWS)], 1) for j in range(COLS)], 2)
        random = (random * 255).clamp(min=0, max=255).to(torch.uint8)
        torchvision.io.write_jpeg(random, "result.jpg")

