"""Continual calibration via temperature scaling
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
On Calibration of Modern Neural Networks.
Adapted from: https://github.com/gpleiss/temperature_scaling
"""
import copy
import torch as th
from torch import nn, optim
from ECE_metrics import ECE
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device, num_bins):
        super(ModelWithTemperature, self).__init__()
        self.model = copy.deepcopy(model)
        self.device = device
        self.num_bins = num_bins
        self.model.eval()
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

    def calibrate(self, lrpp, max_iter, experience_val):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize ExperienceECE.
        experience_val : validation experience
        """
        if self.model:
            for param in self.model.parameters():
                param.requires_grad = False

        # print(self.temperature)
        # for parameters in self.model.parameters():
        #     print(parameters.size())
        #     print(parameters)

        optimizer = optim.LBFGS([self.temperature], lr=lrpp, max_iter=max_iter)
        logits_list = []
        labels_list = []
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_metric = ECE(num_bins=self.num_bins)
        with th.no_grad():
            for input, label, _ in TaskBalancedDataLoader(experience_val):
                logits = self.model(input.to(self.device)).to(self.device)
                logits_list.append(logits)
                labels_list.append(label)
            logits = th.cat(logits_list).to(self.device)
            labels = th.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        ece_metric.update(logits, labels)
        before_temperature_ece_metric = ece_metric.result()
        print('##### Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece_metric))
        ece_metric.reset()

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        ece_metric.update(self.temperature_scale(logits), labels)
        after_temperature_ece_metric = ece_metric.result()
        print('##### Optimal temperature: %.3f' % self.temperature.data)
        print('##### After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece_metric))
        ece_metric.reset()
        if self.model:
            for param in self.model.parameters():
                param.requires_grad = True       
        # print(self.temperature)
        # for parameters in self.model.parameters():
        #     print(parameters.size())
        #     print(parameters)
        return self


class MatrixAndVectorScaling(nn.Module):
    def __init__(self, model, device, num_classes, num_bins, vector_scaling=False):
        super(MatrixAndVectorScaling, self).__init__()
        self.model = copy.deepcopy(model)
        self.device = device
        self.num_bins = num_bins
        self.vector_scaling = vector_scaling
        self.model.eval()
        self.weights = nn.Parameter(th.ones(num_classes, num_classes))
        self.bias = nn.Parameter(th.zeros(num_classes))

    def linear(self, logits):
        if self.vector_scaling:
            return logits.to(self.device) * th.diag(self.weights.to(self.device)) + self.bias.to(self.device)
        else:
            bias = self.bias.unsqueeze(0).expand(logits.size(0), -1)
            return th.matmul(logits.to(self.device), self.weights.to(self.device)) + bias.to(self.device)
        
    def forward(self, input):
        logits = self.model(input)
        return self.linear(logits)

    def calibrate(self, lrpp, max_iter, experience_val):
        if self.model:
            for param in self.model.parameters():
                param.requires_grad = False

        # print(self.weights,self.bias)
        # for parameters in self.model.parameters():
        #     print(parameters.size())
        #     print(parameters)

        optimizer = optim.LBFGS([self.weights, self.bias], lr=lrpp, max_iter=max_iter)
        logits_list = []
        labels_list = []
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_metric = ECE(num_bins=self.num_bins)
        with th.no_grad():
            for input, label, _ in TaskBalancedDataLoader(experience_val):
                logits = self.model(input.to(self.device)).to(self.device)
                logits_list.append(logits)
                labels_list.append(label)
            logits = th.cat(logits_list).to(self.device)
            labels = th.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before scaling
        before_calibration_nll = nll_criterion(logits, labels).item()
        ece_metric.update(logits, labels)
        before_calibration_ece_metric = ece_metric.result()
        print('##### Before calibration - NLL: %.3f, ECE: %.3f' % (before_calibration_nll, before_calibration_ece_metric))
        ece_metric.reset()

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.linear(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after scaling
        after_calibration_nll = nll_criterion(self.linear(logits), labels).item()
        ece_metric.update(self.linear(logits), labels)
        after_calibration_ece_metric = ece_metric.result()
        print('##### After calibration - NLL: %.3f, ECE: %.3f' % (after_calibration_nll, after_calibration_ece_metric))
        ece_metric.reset()
        if self.model:
            for param in self.model.parameters():
                param.requires_grad = True

        # print(self.weights,self.bias)
        # for parameters in self.model.parameters():
        #     print(parameters.size())
        #     print(parameters)

        return self
