import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset


class ChineseWordsDataset(Dataset):
    """
    Chinese Words dataset.
    """

    def __init__(self, opt, transform=None):
        """
        Args:
            opt ([type]): Argument Parser.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_paths = []
        self.labels = []
        self.classes = ['isnull'] + sorted(opt.word_set)
        self.label2idx = {label: i for (i, label) in enumerate(self.classes)}
        for root, dirs, files in os.walk(opt.dataroot):
            for file in files:
                self.img_paths.append(os.path.join(root, file))
                label = re.split('[_.]', file)[-2]
                if label in self.classes:
                    self.labels.append(self.label2idx[label])
                else:
                    self.labels.append(self.label2idx['isnull'])
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
