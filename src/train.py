#!/usr/bin/env python

import numpy as np
import datetime
import time
import torch
import torch.nn as nn
import torchtext
import pytorch_transformers as pt
import csv
from model import SentimentGPT
from utils import Data

# PARAMETERS
EPOCHS = 20
LR = 0.9
MOM = 0.99
DECAY = 0.5

GPU = torch.cuda.is_available()
USE = 0.01
TEST_THR = 100000000

MAX_TWEET_LENGTH = 426


if torch.cuda.is_available() is False:
    GPU = False


tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
cuda = torch.device('cuda')

# Use torchtext datasets as direct conversion to cuda tensors possible
data_fields = [
    ('label', torchtext.data.Field(
        sequential=False,
        use_vocab=False,
        preprocessing=int,
        # is_target=True,
        dtype=torch.Tensor().to(device=cuda)
    )),
    ('tweet', torchtext.data.Field(
        sequential=True,
        fix_length=MAX_TWEET_LENGTH,
        use_vocab=True,
        tokenize=tokenizer.encode,
        pad_token=50256,
        dtype=torch.cuda.LongTensor
    ))]
train_data, val_data, test_data = torchtext.data.TabularDataset.splits(path='../data/', train='train.csv',
                                                                       validation='val.csv', test='test.csv', format='csv', fields=data_fields, csv_reader_params={'delimiter': '|'})

print("DEBUG1")
# Generate torchtext iterator (dataloader equivalent)
train_iterator = torchtext.data.Iterator(train_data, batch_size=128, shuffle=True, device=torch.cuda.current_device(), train=True)
val_iterator = torchtext.data.Iterator(val_data, batch_size=128, shuffle=True, device=torch.cuda.current_device())
test_iterator = torchtext.data.Iterator(test_data, batch_size=128, shuffle=True, device=torch.cuda.current_device())

print("DEBUG2")
model = SentimentGPT(MAX_TWEET_LENGTH)
if GPU:
    model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

print("Start training")
for e in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    len_train_loader = len(train_iterator)
    j = 0
    for i, batch in enumerate(train_iterator):
        start = time.time()
        tweets = batch[0]
        labels = batch[1]
        optimizer.zero_grad()
        output = model(tweets)
        loss = loss_fn(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("Epoch {}: Step {} / {} took {} s".format(e, i, len_train_loader, time.strftime("%H:%M:%S".format(time.time() - start))))
        j += 1
        if j > len_train_loader * USE:
            break
    print("EPOCH: {}, LOSS: {:.2f}".format(e, epoch_loss))
    train_loss = epoch_loss / len(train_iterator)

    val_loss = 0
    for tweets, labels in val_iterator:
        output = model(tweets)
        val_loss += loss_fn(output, labels).item()
    val_loss /= len(val_iterator)
    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))

# Test model for performance. If it exceeds a threshold pickle it and save it on a cloud service
test_loss = 0
for tweets, labels in test_iterator:
    output = model(tweets)
    test_loss += loss_fn(output, labels).item()
test_loss /= len(test_iterator)
print("TEST_LOSS: {:.2f}".format(test_loss))
if test_loss <= TEST_THR:
    print("Performance exceeded threshold. Saving model to models dir to be uploaded to cloud service")
    torch.save(model, '../models/sa_model_{}.pt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
    print("Execute: bash uploadmodels.sh")
