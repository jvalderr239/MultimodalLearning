"""Define Loss Functions for Training"""
import torch.nn as nn

def one_hot_ce_loss(outputs, targets):
    """One hot cross entropy loss for torch data"""
    return nn.CrossEntropyLoss(outputs, targets)
