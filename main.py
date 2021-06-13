import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from lib.dataset import ChineseWordsDataset
from lib.model import Model
from lib.options import Options


def main():
    """
    Training
    """
    opt = Options().parse()

    data_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.726, 0.686, 0.695], [0.205, 0.210, 0.183])
    ])

    dataset = ChineseWordsDataset(opt, data_transform)

    train_size = int(opt.split_rate * len(dataset))
    val_size = len(dataset) - train_size
    opt.dataset_sizes = {
        'train': train_size,
        'val': val_size
    }

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.workers),
        'val': torch.utils.data.DataLoader(val_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=False,
                                           num_workers=opt.workers)
    }

    model_ft = Model(opt)

    model_ft.fit(dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
