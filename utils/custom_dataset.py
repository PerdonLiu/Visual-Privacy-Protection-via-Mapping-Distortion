from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage


__all__ = ['CustomDataset']


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        assert images.shape[0] == labels.shape[0]
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = ToPILImage(mode='RGB')(image)
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)