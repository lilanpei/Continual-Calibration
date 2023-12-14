from torch import nn

class Ent_Loss(nn.Module):
    def __init__(self, ent_weight):
        super(Ent_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.ent_weight = ent_weight

    def forward(self, mb_output, mb_y):
        entropy = 0 #th.distributions.Categorical(mb_output).entropy().mean()

        ent_loss = -self.ent_weight * (entropy if entropy is not None else th.zeros(1))
        loss = self.criterion(mb_output, mb_y) + self.ent_weight * entropy
        return loss
