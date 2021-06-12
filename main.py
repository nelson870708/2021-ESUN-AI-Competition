import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from lib.dataset import ChineseWordsDataset
from lib.model import create_model, train_model
from lib.options import Options


def main():
    """
    Training
    """
    opt = Options().parse()
    if opt.tensorboard:
        opt.writer = SummaryWriter(log_dir=opt.tensorboard_logdir)

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
    opt.class_names = dataset.classes
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    opt.dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.workers),
        'val': torch.utils.data.DataLoader(val_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=False,
                                           num_workers=opt.workers)
    }

    model_ft = create_model(opt)

    opt.criterion = nn.CrossEntropyLoss()

    opt.optimizer = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by cosine function.
    # the first cycle is 5 epochs and the epochs of cycle will increase by multiply 2 after each cycle.
    opt.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt.optimizer, T_0=1, T_mult=2)

    train_model(model_ft, opt)


if __name__ == '__main__':
    main()
