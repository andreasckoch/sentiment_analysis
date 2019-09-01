#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_transformers as pt
import csv
from model import SentimentGPT
from utils import Data

# PARAMETERS
EPOCHS = 1
LR = 0.9
MOM = 0.99
DECAY = 0.0

tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
model = SentimentGPT()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

# DATEN EINLESEN, BATCHEN und dann epochen loopen
with open('data/twitter_sentiment.csv', encoding='latin-1') as file:
    data = csv.reader(file, delimiter='|')
    data = list(data)
"""
[['0',
  "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D",
  '_TheSpecialOne_'],
 ['0',
  "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!",
  'scotthamilton'], 
  ...
"""

split_idx = int(0.8 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

train_data = Data([tokenizer.encode(x[1]) + [50256] * for x in train_data], [x[0] for x in train_data])
val_data = Data([tokenizer.encode(x[1]) for x in val_data], [x[0] for x in val_data])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, num_workers=0)

optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()))

for _ in range(EPOCHS):
    for tweets, labels in train_loader:
        optimizer.zero_grad()
        output = model(tweets)
        loss = loss_fn(output, labels)
        loss.backwards()
        optimizer.step()

    for tweets, labels in val_loader:
        output = model(tweets)
        loss = loss_fn(output, labels)
