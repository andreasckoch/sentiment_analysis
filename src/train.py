#!/usr/bin/env python

import numpy as np
import datetime
import time
import torch
import torch.nn as nn
import torchtext
import pandas as pd
import pytorch_transformers as pt
import csv
from model import SentimentGPT
from utils import Data

# PARAMETERS
EPOCHS = 20
LR = 0.9
MOM = 0.99
DECAY = 0.5

GPU = False
USE = 0.00002
TEST_THR = 100000000

SPLIT_CSV = True


if torch.cuda.is_available() is False:
    GPU = False

# DATEN EINLESEN, BATCHEN und dann epochen loopen
if SPLIT_CSV:
    with open('../data/twitter_sentiment.csv', encoding='latin-1') as file:
        data = csv.reader(file, delimiter='|')
        data = list(data)
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

    max_tweet_len = max([len(x[1]) for x in data])
    print("Max Tweet length: {}".format(max_tweet_len))
    # pad with '<|endoftext|>' = [50256] token such that all tweets have same length
    #data = [x[1] + ['<|endoftext|>'] * (max_tweet_len - len(x[1])) for x in data]  --> pad in data_fields!

    # Split dataset
    split_idx_1 = int(0.8 * len(data))
    split_idx_2 = int(0.9 * len(data))

    with open('../data/train.csv', "w+", encoding='latin-1') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(data[:split_idx_1])
    with open('../data/val.csv', "w+", encoding='latin-1') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(data[split_idx_1:split_idx_2])
    with open('../data/test.csv', "w+", encoding='latin-1') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(data[split_idx_2:])



# tokenize tweets and encode them with the token indices in the GPT2 Token Embedding Dictionary
tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
# tokens = [tokenizer.encode(x[1]) for x in data]
# labels = [torch.tensor(int(x[0])) for x in data]

# Use torchtext datasets as direct conversion to cuda tensors possible
data_fields=[
            ('label', torchtext.data.Field(
                            sequential=False,
                            use_vocab=False,
                            dtype=torch.cuda.ByteTensor
            )),
            ('tweet', torchtext.data.Field(
                            sequential=True,
                            fix_length=max_tweet_len,
                            use_vocab=False,
                            tokenize=tokenizer.encode,
                            pad_token=50256,
                            dtype=torch.cuda.LongTensor
                                 ))]
train_data, val_data, test_data = torchtext.data.TabularDataset.splits(path='../data/', train='train.csv',
                                               validation='val.csv', test='test.csv', format='csv', fields=data_fields)


# Generate torchtext iterator (dataloader equivalent)
train_iterator = torchtext.data.Iterator(train_data, batch_size=128, shuffle=True, device=torch.cuda.current_device())
val_iterator = torchtext.data.Iterator(val_data, batch_size=128, shuffle=True, device=torch.cuda.current_device())
test_iterator = torchtext.data.Iterator(test_data, batch_size=128, shuffle=True, device=torch.cuda.current_device())


model = SentimentGPT(max_tweet_len)
if GPU:
    model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

print("Start training")
for e in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    len_train_loader = len(train_iterator)
    for i, batch in enumerate(train_iterator):
        start = time.time()
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
        print("Epoch {}: Step {} / {} took {} s".format(e, i, len_train_loader, time.strftime("%H:%M:%S".format(time.time() - start))))
    print("EPOCH: {}, LOSS: {:.2f}".format(e, epoch_loss))
    train_loss = epoch_loss / len(train_iterator)

    val_loss = 0
    for tweets, labels in val_iterator:
        if GPU:
            tweets = tweets.cuda()
            labels = labels.cuda()
        output = model(tweets)
        val_loss += loss_fn(output, labels).item()
    val_loss /= len(val_iterator)
    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))

# Test model for performance. If it exceeds a threshold pickle it and save it on a cloud service
test_loss = 0
for tweets, labels in test_iterator:
    if GPU:
        tweets = tweets.cuda()
        labels = labels.cuda()
    output = model(tweets)
    test_loss += loss_fn(output, labels).item()
test_loss /= len(test_iterator)
print("TEST_LOSS: {:.2f}".format(test_loss))
if test_loss <= TEST_THR:
    print("Performance exceeded threshold. Saving model to models dir to be uploaded to cloud service")
    torch.save(model, '../models/sa_model_{}.pt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
    print("Execute: bash uploadmodels.sh")

