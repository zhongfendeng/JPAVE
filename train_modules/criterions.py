#!/usr/bin/env python
# coding: utf-8

import torch


class ClassificationLoss(torch.nn.Module):
    def __init__(self):
        """
        Criterion class, classfication loss
        """
        super(ClassificationLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()



    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        #print('######### logits.device', device)
        loss = self.loss_fn(logits, targets)
        return loss
