import torch
from torch import nn

class Neg_Pearson_MSE(nn.Module):
    def __init__(self, weight=1.0):
        super(Neg_Pearson_MSE, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, labels):
        sum_x = torch.sum(preds, axis=1)
        sum_y = torch.sum(labels, axis=1)
        sum_xy = torch.sum(preds*labels, axis=1)
        sum_x2 = torch.sum(torch.pow(preds, 2), axis=1)
        sum_y2 = torch.sum(torch.pow(labels, 2), axis=1)
        N = preds.shape[1]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        neg_pearson_loss = 1 - pearson
        mse_loss = self.mse_loss(preds, labels)
        loss = self.weight * neg_pearson_loss + (1 - self.weight) * mse_loss
        return loss
