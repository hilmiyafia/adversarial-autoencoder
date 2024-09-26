
import cv2
import numpy
import torch
import torchvision
from model import Autoencoder


ROWS = 4
COLS = 4


if __name__ == "__main__":
    model = Autoencoder().eval()
    model.load_state_dict(torch.load("model.pt", weights_only=False))

    with torch.no_grad():
        random = torch.randn(ROWS*COLS, 256)
        random = model.decoder(random)
        random = torch.concat([torch.concat([random[i+j*ROWS] for i in range(ROWS)], 1) for j in range(COLS)], 2)
        random = (random * 255).clamp(min=0, max=255).to(torch.uint8)
        torchvision.io.write_jpeg(random, "result.jpg")
