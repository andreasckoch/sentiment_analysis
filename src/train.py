#!/usr/bin/env python

import numpy as np
import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytorch_transformers as pt
import csv
from model import SentimentGPT
from utils import collate_wrapper

# PARAMETERS
EPOCHS = 20
LR = 0.9
MOM = 0.99
DECAY = 0.5

GPU = True
USE = 0.02
TEST_THR = 100000000
MAX_TWEET_LEN = 426
LOAD_DATA = True

if torch.cuda.is_available() is False:
    GPU = False

device = None
if GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if LOAD_DATA:
    t = time.time()
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
    train_tokens = torch.tensor([x + [50256] * (MAX_TWEET_LEN - len(x)) for x in train_tokens], device=device)
    val_tokens = torch.tensor([x + [50256] * (MAX_TWEET_LEN - len(x)) for x in val_tokens], device=device)
    test_tokens = torch.tensor([x + [50256] * (MAX_TWEET_LEN - len(x)) for x in test_tokens], device=device)

    # Labels need to be a 1D tensor with integers indicating the class for each value
    train_labels = torch.tensor([int(x[0]) for x in train_data], device=device)
    val_labels = torch.tensor([int(x[0]) for x in val_data], device=device)
    test_labels = torch.tensor([int(x[0]) for x in test_data], device=device)

    train_data = TensorDataset(train_tokens, train_labels)
    val_data = TensorDataset(val_tokens, val_labels)
    test_data = TensorDataset(test_tokens, test_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0)

    print("Finished data preprocessing in {}".format(time.time() - t))

model = SentimentGPT(MAX_TWEET_LEN).to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

print("Start training")
for e in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    len_train_loader = len(train_loader)
    for i, batch in enumerate(train_loader):
        tweets = batch.inp
        labels = batch.tgt
#        if GPU:
#            tweets = tweets.cuda()
#            labels = labels.cuda()
        t = time.time()
        optimizer.zero_grad()
        output = model(tweets)
        loss = loss_fn(output, labels)
        batch_loss = loss.data.cpu().numpy()
        epoch_loss += batch_loss
        loss.backward()
        optimizer.step()
        print("Epoch {}: Step {} / {} took {}s - Train batch loss: {}".format(e, i, len_train_loader, time.time()-t, batch_loss))
    train_loss = epoch_loss / len_train_loader

    val_loss = 0
    len_val_loader = len(val_loader)
    for i, batch in enumerate(val_loader):
        tweets = batch.inp
        labels = batch.tgt
#        if GPU:
#            tweets = tweets.cuda()
#            labels = labels.cuda()
        output = model(tweets)
        batch_loss = loss_fn(output, labels).data.cpu().numpy()
        val_loss += batch_loss
        print("Epoch {} - Val batch loss: {}".format(e, batch_loss))
    val_loss /= len_val_loader
    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))

# Test model for performance. If it exceeds a threshold pickle it and save it on a cloud service
test_loss = 0
len_test_loader = len(test_loader)
for i, batch in enumerate(test_loader):
    tweets = batch.inp
    labels = batch.tgt
#    if GPU:
#        tweets = tweets.cuda()
#        labels = labels.cuda()
    output = model(tweets)
    test_loss += loss_fn(output, labels).data.cpu().numpy()
    print("Step {} / {} - Test batch loss: {}".format(i, len_test_loader, batch_loss))
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

