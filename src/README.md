# Sentiment Analysis on Tweets
This project exists for testing capabilities and characteristics of fine-tuning a sentiment analysis task on 
the Transformer architecture, which was pre-trained on a language modelling task.

## Model
Using the smallest pre-trained GPT-2 Model from the python library pytorch-transformers 
[https://huggingface.co/pytorch-transformers/pretrained_models.html] and adding 2 linear layers on top
for fine-tuning to the classification task.


## Dataset
Training on 1.6 million tweets from [https://www.kaggle.com/kazanova/sentiment140].
Preprocessing was done by only taking tweets and labels into account and discarding non-utf-8 tweets.
For usage, split dataset into train, val and test sets in .csv files and save them locally in a data folder.

## Training
For training, adjust hyperparamters in `train.py` and execute it.
The code is fully hardware agnostic. However, a GPU is recommended as the forward pass of even the smallest 
gpt-2 model with 117M parameters takes 7s on a Nvidia K80 GPU with a batch size of 128 (using full-length 
tweets with `CUT_TWEET_AT=426`). 

