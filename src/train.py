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
USE = 0.0002
TEST_THR = 100000000

if torch.cuda.is_available() is False:
    GPU = False

# DATEN EINLESEN, BATCHEN und dann epochen loopen
with open('../data/twitter_sentiment.csv', encoding='latin-1') as file:
    data = csv.reader(file, delimiter='|')
    data = list(data)
    idx = int(len(data) * USE)
    print("Using {} data points.".format("all" if idx == len(data) else idx))
    data = [data[i] for i in np.random.permutation(len(data))]
    data = data[:idx]


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

# tokenize tweets and encode them with the token indices in the GPT2 Token Embedding Dictionary
tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
tokens = [tokenizer.encode(x[1]) for x in data]
max_tweet_len = max([len(x) for x in tokens])
print("Max Tweet length: {}".format(max_tweet_len))
# pad with '<|endoftext|>' = [50256] token such that all tweets have same length
tokens = [torch.tensor(x + [50256] * (max_tweet_len - len(x))) for x in tokens]
labels = [torch.tensor(int(x[0])) for x in data]


# Create Data Loaders for Training and Validation Data
split_idx_1 = int(0.8 * len(data))
split_idx_2 = int(0.9 * len(data))
train_data = Data(tokens[:split_idx_1], labels[:split_idx_1])
val_data = Data(tokens[split_idx_1:split_idx_2], labels[split_idx_1:split_idx_2])
test_data = Data(tokens[split_idx_2:], labels[split_idx_2:])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0)


model = SentimentGPT(max_tweet_len)
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
            tweets = tweets.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(tweets)
        loss = loss_fn(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
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


#!/usr/bin/env python


# PARAMETERS
EPOCHS = 20
LR = 0.9
MOM = 0.99
DECAY = 0.5

GPU = True
USE = 1.0  # 0.0002
TEST_THR = 100000000

if torch.cuda.is_available() is False:
    GPU = False

# DATEN EINLESEN, BATCHEN und dann epochen loopen
with open('../data/twitter_sentiment.csv', encoding='latin-1') as file:
    data = csv.reader(file, delimiter='|')
    data = list(data)
    USE = USE if USE <= 1.0 else 1.0
    idx = int(len(data) * USE)
    print("Using {} data points.".format("all" if idx == len(data) else idx))
    data = [data[i] for i in np.random.permutation(len(data))]
    data = data[:idx]

"""
[['0', 
  "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D",
  '_TheSpecialOne_'],
 ['0',
  "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!",
  'scotthamilton'], 
  ...
"""

# tokenize tweets and encode them with the token indices in the GPT2 Token Embedding Dictionary
tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
tokens = [tokenizer.encode(x[1]) for x in data]
max_tweet_len = max([len(x) for x in tokens])
print("Max Tweet length: {}".format(max_tweet_len))
# pad with '<|endoftext|>' = [50256] token such that all tweets have same length
tokens = [torch.tensor(x + [50256] * (max_tweet_len - len(x))) for x in tokens]
labels = [torch.tensor(int(x[0])) for x in data]


# Create Data Loaders for Training and Validation Data
split_idx_1 = int(0.8 * len(data))
split_idx_2 = int(0.9 * len(data))
train_data = Data(tokens[:split_idx_1], labels[:split_idx_1])
val_data = Data(tokens[split_idx_1:split_idx_2], labels[split_idx_1:split_idx_2])
test_data = Data(tokens[split_idx_2:], labels[split_idx_2:])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0)


model = SentimentGPT(max_tweet_len)
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
        tt = time.time()
        t = time.time()
        tweets = batch[0]
        labels = batch[1]
        if GPU:
            tweets = tweets.cuda()
            labels = labels.cuda()
        print("move to cuda: {:.4f}s".format(time.time() - t))
        optimizer.zero_grad()
        t = time.time()
        output = model(tweets)
        print("push tweets through model: {:.4f}s".format(time.time() - t))
        t = time.time()
        loss = loss_fn(output, labels)
        print("loss calc: {:.4f}s".format(time.time() - t))
#        if i%2==0:
#            t = time.time()
#            epoch_loss += loss.item()
#            print("add loss.item to epoch_loss: {:.4f}s".format(time.time()-t))
        t = time.time()
        loss.backward()
        print("backprop: {:.4f}s".format(time.time() - t))
        t = time.time()
        optimizer.step()
        print("optim step: {:.4f}s".format(time.time() - t))

        print("Epoch {}: Step {} / {}, batch time: {:.4f}s".format(e, i, len_train_loader, time.time() - tt))
    print("EPOCH: {}, LOSS: {:.2f}".format(e, epoch_loss))
    train_loss = epoch_loss / len(train_loader)

#    val_loss = 0
#    for tweets, labels in val_loader:
#        if GPU:
#            tweets = tweets.cuda()
#            labels = labels.cuda()
#        output = model(tweets)
#        val_loss += loss_fn(output, labels).item()
#    val_loss /= len(val_loader)
#    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))

# Test model for performance. If it exceeds a threshold pickle it and save it on a cloud service
#test_loss = 0
# for tweets, labels in test_loader:
#    if GPU:
#        tweets = tweets.cuda()
#        labels = labels.cuda()
#    output = model(tweets)
#    test_loss += loss_fn(output, labels).item()
#test_loss /= len(test_loader)
#print("TEST_LOSS: {:.2f}".format(test_loss))
# if test_loss <= TEST_THR:
#    print("Performance exceeded threshold. Saving model to models dir to be uploaded to cloud service")
#    torch.save(model, '../models/sa_model_{}.pt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
#    print("Execute: bash uploadmodels.sh")
