""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

"""

import argparse
import os

import torch


class Options:
    """
    Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--dataroot', default='./data/clean v2', help='path to dataset')
        self.parser.add_argument('--split_rate', type=int, default=0.8,
                                 help='split training and valid data by the split rate')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
        self.parser.add_argument('--img_size', type=int, default=64, help='input image size.')
        self.parser.add_argument('--model_name', type=str, default='efficientnet-b0',
                                 help='chooses which efficientnet to use.')
        self.parser.add_argument('--outfile', default='./output', help='folder to output model checkpoints')
        self.parser.add_argument('--load_weights_path', default='', help='path to checkpoints (to continue training)')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='start from epoch i')
        self.parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs to train for')
        self.parser.add_argument('--tensorboard', action='store_true', help='output tensorboard log')
        self.parser.add_argument('--tensorboard_logdir', type=str, default='runs',
                                 help='only work when "tensorboard" is true')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        official_dict = './data/training data dic.txt'
        with open(official_dict, encoding='utf-8') as f:
            self.word_set = f.read().split()

        self.opt = None

    def parse(self):
        """
        Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        self.opt.device = self.device
        self.opt.word_set = self.word_set

        if not os.path.isdir(self.opt.outfile):
            os.makedirs(self.opt.outfile)

        return self.opt
