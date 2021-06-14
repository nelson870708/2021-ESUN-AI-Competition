import os
import re

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lib.dataset import ChineseWordsDataset
from lib.model import Model
from lib.options import Options


def main():
    """
    Training
    """
    opt = Options().parse()

    # prepared data
    img_paths = []
    labels = []
    classes = ['isnull'] + sorted(opt.word_set)
    label2idx = {label: i for (i, label) in enumerate(classes)}

    for root, dirs, files in os.walk(opt.dataroot):
        for file in files:
            img_paths.append(os.path.join(root, file))
            label = re.split('[_.]', file)[-2]
            if label in classes:
                labels.append(label2idx[label])
            else:
                labels.append(label2idx['isnull'])

    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(img_paths, labels, shuffle=True,
                                                                                train_size=opt.split_rate)
    img_paths = {'train': train_img_paths, 'val': val_img_paths}
    labels = {'train': train_labels, 'val': val_labels}

    # data augmentation and normalize
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(opt.img_size, scale=(0.9, 1.0)),
            transforms.RandomAffine(15, (0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.726, 0.686, 0.695], [0.205, 0.210, 0.183])
        ]),
        'val': transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.726, 0.686, 0.695], [0.205, 0.210, 0.183])
        ]),
    }

    image_datasets = {x: ChineseWordsDataset(img_paths[x], labels[x], data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.workers) for x in ['train', 'val']}

    # create model and start training
    model_ft = Model(opt)

    model_ft.fit(dataloaders)


if __name__ == '__main__':
    main()
