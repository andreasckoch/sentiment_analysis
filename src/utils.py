#!/usr/bin/env python

import torch

"""
Map-style dataset as an iteration can be well performed randomly.
See https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading 
"""


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


"""
Create function to apply to batch to allow for memory pinning when using a custom batch/custom dataset.
Following guide on https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading
"""


class SimpleCustomBatch:

    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def get_hms_string(s):
    m = s // 60
    s %= 60
    h = m // 60
    m %= 60
    return "{}h:{}min:{:.1f}s".format(int(h), int(m), s)
