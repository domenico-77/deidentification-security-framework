import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class LFWDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        tensor_img = self.transform(img) * 255.0  # Scala [0, 255]
        return tensor_img, path