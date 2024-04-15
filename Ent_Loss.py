import torch
from torch import nn


class Ent_Loss(nn.Module):
    def __init__(self, ent_weight):
        super(Ent_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.ent_weight = ent_weight

    def forward(self, mb_output, mb_y):
        cross_entropy = torch.nn.functional.cross_entropy(mb_output, mb_y)
        probs = torch.softmax(mb_output, dim=-1)
        log_probs = torch.log_softmax(mb_output, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return cross_entropy - self.ent_weight * entropy
