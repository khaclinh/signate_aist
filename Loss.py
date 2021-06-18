import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        F_loss = at * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean()
