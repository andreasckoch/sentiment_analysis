#!/usr/bin/env python

import torch


class Data(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        tweet = self.X[index]
        label = self.y[index]

        return tweet, label

    def __len__(self):
        return len(self.y)
