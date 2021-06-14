import torch
from PIL import Image
from torch.utils.data import Dataset


class ChineseWordsDataset(Dataset):
    """
    Chinese Words dataset.
    """

    def __init__(self, img_paths, labels, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.img_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
