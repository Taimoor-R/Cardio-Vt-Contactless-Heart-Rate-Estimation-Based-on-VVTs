import torch
from torch import nn
class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):
        sum_x = torch.sum(preds, axis=1)
        sum_y = torch.sum(labels, axis=1)
        sum_xy = torch.sum(preds*labels, axis=1)
        sum_x2 = torch.sum(torch.pow(preds, 2), axis=1)
        sum_y2 = torch.sum(torch.pow(labels, 2), axis=1)
        N = preds.shape[1]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        loss = 1 - pearson
        loss = torch.mean(loss)
        return loss
