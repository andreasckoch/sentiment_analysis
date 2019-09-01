#!/usr/bin/env python

import torch.nn as nn
import pytorch_transformers as pt


class SentimentGPT(nn.Module):
    """
    A simple sentiment analysis task (positive/negative) fine-tuned on the GPT2 Model. Dataset used from Kaggle:
    https://www.kaggle.com/kazanova/sentiment140
    """

    def __init__(self, tweet_len):
        super(SentimentGPT, self).__init__()

        self.tweet_len = tweet_len

        self.gpt2_stem = pt.modeling_gpt2.GPT2Model.from_pretrained('gpt2')
        self.collapse_tweet = nn.Linear(in_features=self.tweet_len, out_features=1, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2))

        # no gradient in gpt2
        for param in self.gpt2_stem.parameters():
            param.requires_grad = False

        # initialize weights of linear layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.gpt2_stem(x)[0]
        x = x.transpose(1, 2)
        x = self.collapse_tweet(x)
        x = x.squeeze(2)
        x = self.classifier(x)
        return x  # then cross entropy loss
