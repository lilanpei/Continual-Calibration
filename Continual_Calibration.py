"""Continual calibration via temperature scaling
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
On Calibration of Modern Neural Networks.
Adapted from: https://github.com/gpleiss/temperature_scaling
"""
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from avalanche.training.supervised import Naive, JointTraining, Replay
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from ECE_metrics import ExperienceECE, ExpECEHistogram, ECE
from ModelWithTemperature import ModelWithTemperature
from _ECELoss import _ECELoss

class Continual_Calibration:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 strategy_name,
                 benchmark,
                 train_mb_size,
                 train_epochs,
                 mem_size,
                 eval_mb_size,
                 eval_plugin,
                 device,
                 pp_calibration_mode
                 ):
        self.model = model
        self.strategy_name = strategy_name
        self.benchmark = benchmark
        self.mem_size = mem_size
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.eval_mb_size = eval_mb_size
        self.device = device
        self.eval_plugin = eval_plugin
        self.optimizer = optimizer
        self.criterion = criterion
        print("@@@@@@@@@@", criterion, "@@@@@@@@@@")
        self.pp_calibration_mode = pp_calibration_mode

        if self.strategy_name == "JointTraining":
            self.strategy = JointTraining(
                self.model,
                optimizer,
                self.criterion,
                train_mb_size=self.train_mb_size,
                train_epochs=self.train_epochs,
                device=self.device
            )
        else:
            if self.strategy_name == "Replay":
                self.strategy = Replay(
                    self.model,
                    self.optimizer,
                    mem_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.train_mb_size,
                    train_epochs=self.train_epochs,
                    eval_mb_size=self.eval_mb_size,
                    evaluator=self.eval_plugin,
                    device=self.device
                    )
            else:
                self.strategy = Naive(
                    self.model,
                    self.optimizer,
                    criterion=self.criterion,
                    train_mb_size=self.train_mb_size,
                    train_epochs=self.train_epochs,
                    eval_mb_size=self.eval_mb_size,
                    evaluator=self.eval_plugin,
                    device=self.device
                    )

    # TRAINING LOOP
    def train(self,):
            print('Starting experiment...')
            results = []
            if self.strategy_name == "JointTraining":
                self.strategy.train(self.benchmark.train_stream)
                results.append(self.strategy.eval(self.benchmark.test_stream))
            else:
                for experience_tr, experience_val in zip(self.benchmark.train_stream, self.benchmark.valid_stream):
                    print("Start of experience: ", experience_tr.current_experience)
                    print("Current Classes: ", experience_tr.classes_in_this_experience)

                    # train returns a dictionary which contains all the metric values
                    self.strategy.train(experience_tr)
                    print('Training completed')

                    if self.pp_calibration_mode:
                        self.model = ModelWithTemperature(self.model, self.device)
                        print("%%%% before calibrate temperature", self.model.temperature.data)
                        self.calibrate_temperature(experience_val)
                        optimal_temperature = self.model.temperature
                        print("%%%% after calibrate temperature", optimal_temperature.data)

                    print('Computing accuracy on the whole test set')
                    # test also returns a dictionary which contains all the metric values
                    results.append(self.strategy.eval(self.benchmark.test_stream))
                    
                    if self.pp_calibration_mode:
                        self.model = self.model.model

            return results

    def calibrate_temperature(self, experience_val):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize ExperienceECE.
        experience_val : validation experience
        """

        optimizer = optim.LBFGS([self.model.temperature], lr=0.01, max_iter=50)
        logits_list = []
        labels_list = []
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)
        ece_metric = ECE()
        with th.no_grad():
            for input, label, _ in TaskBalancedDataLoader(experience_val.dataset):
                logits = self.model(input.to(self.device)).to(self.device)
                logits_list.append(logits)
                labels_list.append(label)
            logits = th.cat(logits_list).to(self.device)
            labels = th.cat(labels_list).to(self.device)
        
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        ece_metric.update(logits, labels)
        before_temperature_ece_metric = ece_metric.result()
        print('##### Before temperature - NLL: %.3f, ECE: %.3f, ECE_Metric: %.3f' % (before_temperature_nll, before_temperature_ece, before_temperature_ece_metric))
        ece_metric.reset()

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.model.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.model.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.model.temperature_scale(logits), labels).item()
        ece_metric.update(self.model.temperature_scale(logits), labels)
        after_temperature_ece_metric = ece_metric.result()
        print('##### Optimal temperature: %.3f' % self.model.temperature.data)
        print('##### After temperature - NLL: %.3f, ECE: %.3f, ECE_Metric: %.3f' % (after_temperature_nll, after_temperature_ece, after_temperature_ece_metric))
        ece_metric.reset()

        return self
