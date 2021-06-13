import copy
import os.path
import time

import torch
import torch.nn as nn
from PIL.Image import Image
from efficientnet_pytorch import EfficientNet
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Model:
    def __init__(self, opt):
        self.opt = opt

        self.model = EfficientNet.from_pretrained(opt.model_name, num_classes=opt.num_class)
        self.model = self.model.to(device=opt.device)
        if opt.load_weights_path:
            self.model.load_state_dict(torch.load(opt.load_weights_path))

        self.criterion = nn.CrossEntropyLoss()

        if opt.optimizer is 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr)

        if opt.lr_scheduler is 'CosineAnnealingWarmRestarts':
            self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2)
        else:
            self.lr_scheduler = None

        if opt.tensorboard:
            self.writer = SummaryWriter(log_dir=opt.tensorboard_logdir)

    def fit(self, train_dataloader, val_dataloader):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        hist = []

        for epoch in range(self.opt.start_epoch, self.opt.start_epoch + self.opt.n_epoch):
            print(f'Epoch {epoch}/{self.opt.start_epoch + self.opt.n_epoch - 1}')
            print('-' * 10)

            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = train_dataloader
                else:
                    self.model.eval()  # Set model to evaluate mode
                    dataloader = val_dataloader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.opt.device)
                    labels = labels.to(self.opt.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            if self.lr_scheduler:
                                self.lr_scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.opt.dataset_sizes[phase]
                epoch_acc = running_corrects / self.opt.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss: .4f} Acc: {epoch_acc: .4f}')

                if phase == 'train':
                    train_loss = epoch_loss
                    train_acc = epoch_acc
                else:
                    val_loss = epoch_loss
                    val_acc = epoch_acc
                    if val_acc > best_acc:
                        # deep copy the model
                        best_acc = val_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(best_model_wts, os.path.join(self.opt.outfile, self.opt.model_name + '.pth'))

            hist.append({'train loss': train_loss,
                         'train accuracy': train_acc,
                         'valid loss': val_loss,
                         'valid accuracy': val_acc})

            if self.opt.tensorboard:
                self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                self.writer.flush()
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60: .0f}m {time_elapsed % 60: .0f}s')
        print(f'Best val Acc: {best_acc: 4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if self.opt.tensorboard:
            self.writer.close()

        return hist

    def predict(self, test_dataloader):
        return
