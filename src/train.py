#!/usr/bin/env python

import numpy as np
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
DECAY = 0.0

USE = 0.0002


# DATEN EINLESEN, BATCHEN und dann epochen loopen
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

# tokenize tweets and encode them with the token indices in the GPT2 Token Embedding Dictionary
tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
tokens = [tokenizer.encode(x[1]) for x in data]
max_tweet_len = max([len(x) for x in tokens])
print("Max Tweet length: {}".format(max_tweet_len))
# pad with '<|endoftext|>' = [50256] token such that all tweets have same length
tokens = [torch.tensor(x + [50256] * (max_tweet_len - len(x))) for x in tokens]
labels = [torch.tensor(int(x[0])) for x in data]


# Create Data Loaders for Training and Validation Data
split_idx = int(0.8 * len(data))
train_data = Data(tokens[:split_idx], labels[:split_idx])
val_data = Data(tokens[split_idx:], labels[split_idx:])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=True, num_workers=0)


model = SentimentGPT(max_tweet_len)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

for e in range(EPOCHS):
    epoch_loss = 0
    for tweets, labels in train_loader:
        optimizer.zero_grad()
        output = model(tweets)
        loss = loss_fn(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("EPOCH: {}, LOSS: {:.2f}".format(e, epoch_loss))

    # for tweets, labels in val_loader:
    #     output = model(tweets)
    #     loss = loss_fn(output, labels)
