#!/usr/bin/env python

import numpy as np
import datetime
import time
import torch
import torch.nn as nn
import pytorch_transformers as pt
import csv
from model import SentimentGPT
from utils import Data

# PARAMETERS
EPOCHS = 20
LR = 0.9
MOM = 0.99
DECAY = 0.5

GPU = True
USE = 0.02
TEST_THR = 100000000
MAX_TWEET_LEN = 426

if torch.cuda.is_available() is False:
    GPU = False

with open('../data/train.csv', encoding='latin-1') as file:
    train_data = csv.reader(file, delimiter='|')
    train_data = list(train_data)
    idx = int(len(train_data) * USE)
    print("Using {} data points.".format("all" if idx == len(train_data) else idx))
    train_data = train_data[:idx]
with open('../data/val.csv', encoding='latin-1') as file:
    val_data = csv.reader(file, delimiter='|')
    val_data = list(val_data)
    idx = int(len(val_data) * USE)
    val_data = val_data[:idx]
with open('../data/test.csv', encoding='latin-1') as file:
    test_data = csv.reader(file, delimiter='|')
    test_data = list(test_data)
    idx = int(len(test_data) * USE)
    test_data = test_data[:idx]

tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
train_tokens = [tokenizer.encode(x[1]) for x in train_data]
val_tokens = [tokenizer.encode(x[1]) for x in val_data]
test_tokens = [tokenizer.encode(x[1]) for x in test_data]
# pad with '<|endoftext|>' = [50256] token such that all tweets have same length
train_tokens = [torch.tensor(x + [50256] * (MAX_TWEET_LEN - len(x))) for x in train_tokens]
val_tokens = [torch.tensor(x + [50256] * (MAX_TWEET_LEN - len(x))) for x in val_tokens]
test_tokens = [torch.tensor(x + [50256] * (MAX_TWEET_LEN - len(x))) for x in test_tokens]
train_labels = [torch.tensor(int(x[0])) for x in train_data]
val_labels = [torch.tensor(int(x[0])) for x in val_data]
test_labels = [torch.tensor(int(x[0])) for x in test_data]

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0)


model = SentimentGPT(MAX_TWEET_LEN)
if GPU:
    model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

print("Start training")
for e in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    len_train_loader = len(train_loader)
    for i, batch in enumerate(train_loader):
        tweets = batch[0]
        labels = batch[1]
        if GPU:
            t = time.time()
            tweets = tweets.cuda()
            labels = labels.cuda()
            print("load on cuda: {:.4f}".format(time.time() - t))
        t = time.time()
        optimizer.zero_grad()
        print("load on cuda: {:.4f}".format(time.time() - t))
        t = time.time()
        output = model(tweets)
        print("pass through model: {:.4f}".format(time.time() - t))
        t = time.time()
        loss = loss_fn(output, labels)
        print("loss calc: {:.4f}".format(time.time() - t))
        t = time.time()
        epoch_loss += loss.item()
        print("add loss item: {:.4f}".format(time.time() - t))
        t = time.time()
        loss.backward()
        print("loss backward: {:.4f}".format(time.time() - t))
        t = time.time()
        optimizer.step()
        print("optimizer step: {:.4f}".format(time.time() - t))

    print("Epoch {}: Step {} / {}".format(e, i, len_train_loader))
    print("EPOCH: {}, LOSS: {:.2f}".format(e, epoch_loss))
    train_loss = epoch_loss / len(train_loader)

    val_loss = 0
    for tweets, labels in val_loader:
        if GPU:
            tweets = tweets.cuda()
            labels = labels.cuda()
        output = model(tweets)
        val_loss += loss_fn(output, labels).item()
    val_loss /= len(val_loader)
    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))

# Test model for performance. If it exceeds a threshold pickle it and save it on a cloud service
test_loss = 0
for tweets, labels in test_loader:
    if GPU:
        tweets = tweets.cuda()
        labels = labels.cuda()
    output = model(tweets)
    test_loss += loss_fn(output, labels).item()
test_loss /= len(test_loader)
print("TEST_LOSS: {:.2f}".format(test_loss))
if test_loss <= TEST_THR:
    print("Performance exceeded threshold. Saving model to models dir to be uploaded to cloud service")
    torch.save(model, '../models/sa_model_{}.pt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
    print("Execute: bash uploadmodels.sh")



"""
USE=0.1, batch_size=128:
Epoch 0: Step 46 / 997
move to cuda: 0.0008s
push tweets through model: 0.0159s
loss calc: 7.3934s
backprop: 0.0009s
optim step: 0.0004s

class CrossEntropyLoss(_WeightedLoss):
    def forward(self, input, target):
        return cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
def cross_entropy(input, target):
  return nll_loss(log_softmax(input, 1), target)

def log_softmax(input, dim=None):
  return input.log_softmax(dim, dtype=dtype)

def nll_loss(input, target):
  return torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)



"""

