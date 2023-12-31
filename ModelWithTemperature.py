"""Continual calibration via temperature scaling
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
On Calibration of Modern Neural Networks.
Adapted from: https://github.com/gpleiss/temperature_scaling
"""
import torch as th
from torch import nn

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(th.ones(1)) # * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
        return logits / temperature
