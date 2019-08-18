#!/usr/bin/env python

class Data(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        tweet = self.X[index]
        label = self.y[index]

        return img, label

    def __len__(self):
        return len(self.y)