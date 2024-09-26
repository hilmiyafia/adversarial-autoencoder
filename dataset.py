
import os
import torch
import torchvision.transforms.v2


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
        self.transform = torch.nn.Sequential(
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.RandomHorizontalFlip(),
            torchvision.transforms.v2.Resize(64))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.files[index])
        return self.transform(image)
        
