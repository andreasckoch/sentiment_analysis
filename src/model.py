#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_transformers as pt

class SentimentGPT(nn.Module):
    """
    A simple sentiment analysis task (positive/negative) fine-tuned on the GPT2 Model. Dataset used from Kaggle:
    https://www.kaggle.com/kazanova/sentiment140
    """

    def __init__(self):
        super(SentimentGPT, self).__init__()

        gpt2_stem = pt.modeling_gpt2.GPT2Model.from_pretrained('gpt2')
        classifier = nn.Sequential(
            nn.Linear(nn.Linear(in_features=768, out_features=256, bias=True),
                      nn.ReLU(),
                      nn.Linear(nn.Linear(in_features=265, out_features=2))

            # no gradient in gpt2
            for param in gpt2_stem.parameters():
                param.requires_grad = False

        # initialize weights of linear layers
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def forward(x):
        x = self.gpt2_stem(x)
        x = self.classifier(x)
        return x  # dann cross entropy loss

