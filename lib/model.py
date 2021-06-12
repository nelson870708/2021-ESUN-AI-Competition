import copy
import os.path
import time

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
from tqdm import tqdm


def create_model(opt):
    model = EfficientNet.from_pretrained(opt.model_name)
    num_ftrs = model._fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model._fc = nn.Linear(num_ftrs, len(opt.class_names))

    model = model.to(opt.device)

    if opt.load_weights_path:
        model.load_state_dict(torch.load(opt.load_weights_path))
    return model

def train_model(model, opt):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    end_epoch = opt.n_epoch + opt.start_epoch - 1
    for epoch in range(opt.start_epoch, end_epoch + 1):
        print(f'Epoch {epoch}/{end_epoch}')
        print('-' * 10)

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(opt.dataloaders[phase]):
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                # zero the parameter gradients
                opt.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = opt.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        opt.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and opt.scheduler:
                opt.scheduler.step()

            epoch_loss = running_loss / opt.dataset_sizes[phase]
            epoch_acc = running_corrects / opt.dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss: .4f} Acc: {epoch_acc: .4f}')

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc

            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(opt.outfile, opt.model_name + '.pth'))

        if opt.tensorboard:
            opt.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            opt.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            opt.writer.flush()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60: .0f}m {time_elapsed % 60: .0f}s')
    print(f'Best val Acc: {best_acc: 4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    if opt.tensorboard:
        opt.writer.close()
    return model
