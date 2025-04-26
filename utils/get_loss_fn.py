import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_fn(loss_fn):
    if loss_fn == 'mse':
        return nn.MSELoss()