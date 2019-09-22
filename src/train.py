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
from utils import get_hms_string


# H-PARAMETERS
EPOCHS = 2
LR = 0.9
MOM = 0.99
DECAY = 0.5

# technical parameters
GPU = True
USE = 0.1  # factor indicating amount of the dataset used
MAX_TWEET_LEN = 426  # previously measured in dataset, leave unchanged
CUT_TWEETS_AT = 100  # cut the tweets at a particular length
LOAD_DATA = True
DEBUG = False


device = None
if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if LOAD_DATA:
    t = time.time()

    # Dataset used: https://www.kaggle.com/kazanova/sentiment140
    with open('../data/train.csv', encoding='latin-1') as file:
        train_data = csv.reader(file, delimiter='|')
        train_data = list(train_data)
        idx = int(len(train_data) * USE)
        print("Using {} data points ({}%).".format(
            "all" if idx == len(train_data) else idx, USE * 100))
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

    # tokenize tweets (or rather encode to indices of tokens in dictionary of gpt2)
    tokenizer = pt.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
    train_tokens = [tokenizer.encode(x[1]) for x in train_data]
    val_tokens = [tokenizer.encode(x[1]) for x in val_data]
    test_tokens = [tokenizer.encode(x[1]) for x in test_data]

    # pad with '<|endoftext|>' = [50256] token, then cut lenght such that all tweets have same length
    train_tokens = [(x + [50256] * (MAX_TWEET_LEN - len(x)))[:CUT_TWEETS_AT] for x in train_tokens]
    val_tokens = [(x + [50256] * (MAX_TWEET_LEN - len(x)))[:CUT_TWEETS_AT] for x in val_tokens]
    test_tokens = [(x + [50256] * (MAX_TWEET_LEN - len(x)))[:CUT_TWEETS_AT] for x in test_tokens]
    MAX_TWEET_LEN = CUT_TWEETS_AT

    """ Dataset is small enough to completely store on the gpu. 
    If that's not possible, move batches to gpu while iterating over them and set pin_memory=True, num_workers=8 (ca.) in data loaders.
    See https://pytorch.org/docs/master/notes/cuda.html as well as link in 'utils.py' for details on memory pinning and time measurement. """
    train_tokens = torch.tensor(train_tokens, device=device)
    val_tokens = torch.tensor(val_tokens, device=device)
    test_tokens = torch.tensor(test_tokens, device=device)

    # Labels need to be a 1D tensor with integers indicating the class for each value
    train_labels = torch.tensor([int(x[0]) for x in train_data], device=device)
    val_labels = torch.tensor([int(x[0]) for x in val_data], device=device)
    test_labels = torch.tensor([int(x[0]) for x in test_data], device=device)

    # wrap dataset and create DataLoader
    train_data = TensorDataset(train_tokens, train_labels)
    val_data = TensorDataset(val_tokens, val_labels)
    test_data = TensorDataset(test_tokens, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=256, shuffle=True, num_workers=0)

    print("Finished data preprocessing in {}".format(time.time() - t))

model = SentimentGPT(MAX_TWEET_LEN).to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.classifier.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)

print("Start training for {} epochs.".format(EPOCHS))
for e in range(EPOCHS):
    epoch_start = time.time()
    train_loss = 0
    len_train_loader = len(train_loader)
    batch_times, i = [], 0
    for tweets, labels in train_loader:
        t = time.time()
        if DEBUG and torch.cuda.is_available():
            # measure the time the operations take to calculate

            start_optim_zero = torch.cuda.Event(enable_timing=True)
            end_optim_zero = torch.cuda.Event(enable_timing=True)
            start_model = torch.cuda.Event(enable_timing=True)
            end_model = torch.cuda.Event(enable_timing=True)
            start_loss = torch.cuda.Event(enable_timing=True)
            end_loss = torch.cuda.Event(enable_timing=True)
            start_loss_item = torch.cuda.Event(enable_timing=True)
            end_loss_item = torch.cuda.Event(enable_timing=True)
            start_loss_back = torch.cuda.Event(enable_timing=True)
            end_loss_back = torch.cuda.Event(enable_timing=True)
            start_optim_step = torch.cuda.Event(enable_timing=True)
            end_optim_step = torch.cuda.Event(enable_timing=True)

            start_optim_zero.record()
            optimizer.zero_grad()
            end_optim_zero.record()
            start_model.record()
            output = model(tweets)
            end_model.record()
            start_loss.record()
            loss = loss_fn(output, labels)
            end_loss.record()
            start_loss_item.record()
            batch_loss = loss.item()
            train_loss += batch_loss
            end_loss_item.record()
            start_loss_back.record()
            loss.backward()
            end_loss_back.record()
            start_optim_step.record()
            optimizer.step()
            end_optim_step.record()

            torch.cuda.synchronize()
            t_optim_zero = start_optim_zero.elapsed_time(end_optim_zero)
            t_model = start_model.elapsed_time(end_model)
            t_loss = start_loss.elapsed_time(end_loss)
            t_loss_item = start_loss_item.elapsed_time(end_loss_item)
            t_loss_back = start_loss_back.elapsed_time(end_loss_back)
            t_optim_step = start_optim_step.elapsed_time(end_optim_step)
            print("DEBUG TIMES in ms\nOptim Zero: {}\nModel: {}\nLoss: {}\nLoss Item: {}\nLoss Back: {}\nOptim Step: {}"
                  .format(t_optim_zero, t_model, t_loss, t_loss_item, t_loss_back, t_optim_step))

        else:
            optimizer.zero_grad()
            output = model(tweets)
            loss = loss_fn(output, labels)
            batch_loss = loss.item()
            train_loss += batch_loss
            loss.backward()
            optimizer.step()

        batch_times.append(time.time() - t)
        total_time_left = np.mean(
            batch_times) * (len_train_loader - i + (EPOCHS - e) * len_train_loader)
        print("Epoch {}: Step {} / {} took {}s (~{} left) - Train batch loss: {}".format(
            e, i, len_train_loader, batch_times[-1], get_hms_string(total_time_left), batch_loss))
        i += 1

    train_loss /= len_train_loader
    val_loss = 0

    # ## To finetune h-parameters, one can use this validation train loop
    # len_val_loader = len(val_loader)
    # for i, batch in enumerate(val_loader):
    #     t = time.time()
    #     tweets = batch[0]
    #     labels = batch[1]
    #     output = model(tweets)
    #     batch_loss = loss_fn(output, labels).item()
    #     val_loss += batch_loss
    #     print("Epoch {}: Step {} / {} took {}s - Val batch loss: {}".format(
    #           e, i, len_val_loader, time.time() - t, batch_loss))
    # val_loss /= len_val_loader

    print("EPOCH: {} took {}, TRAIN_LOSS: {:.2f}, VAL_LOSS: {:.2f}".format(
        e, time.strftime("%H:%M:%S".format(time.time() - epoch_start)), train_loss, val_loss))
    print('#' * 65)

# Test model for performance and then save it.
test_loss = 0
len_test_loader = len(test_loader)
for i, batch in enumerate(test_loader):
    tweets = batch[0]
    labels = batch[1]
    output = model(tweets)
    test_loss += loss_fn(output, labels).item()
    print("Step {} / {} - Test batch loss: {}".format(i, len_test_loader, batch_loss))
test_loss /= len(test_loader)
print("TEST_LOSS: {:.2f}".format(test_loss))
torch.save(model, '../models/sa_model_{}.pt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
