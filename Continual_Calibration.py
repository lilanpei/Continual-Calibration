"""Continual calibration via temperature scaling
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
On Calibration of Modern Neural Networks.
Adapted from: https://github.com/gpleiss/temperature_scaling
"""
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from avalanche.training.supervised import Naive, JointTraining, Replay
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset, AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_attribute import ConstantSequence
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from ECE_metrics import ExperienceECE, ExpECEHistogram, ECE
from ModelWithTemperature import ModelWithTemperature
from typing import Iterable
import numpy as np


class Continual_Calibration:
    def __init__(self,
                 tb_logger,
                 model,
                 optimizer,
                 plugins,
                 criterion,
                 strategy_name,
                 benchmark,
                 train_mb_size,
                 train_epochs,
                 mem_size,
                 eval_mb_size,
                 eval_plugin,
                 device,
                 pp_calibration_mode,
                 pp_cal_mixed_data,
                 calibration_mode_str,
                 logdir
                 ):
        self.tb_logger = tb_logger
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
        self.pp_calibration_mode = pp_calibration_mode
        self.pp_cal_mixed_data = pp_cal_mixed_data
        self.calibration_mode_str = calibration_mode_str
        self.log_dir = logdir

        if self.strategy_name == "JointTraining":
            self.strategy = JointTraining(
                self.model,
                optimizer,
                self.criterion,
                train_mb_size=self.train_mb_size,
                train_epochs=self.train_epochs,
                eval_mb_size=self.eval_mb_size,
                evaluator=self.eval_plugin,
                plugins=plugins,
                eval_every=1,
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
                    plugins=plugins,
                    eval_every=1,
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
                    plugins=plugins,
                    eval_every=1,
                    device=self.device
                    )

    # TRAINING LOOP
    def train(self,):
            print('Starting experiment...')
            results = []
            val_experiences_list = []
            for exp in self.benchmark.valid_stream:
                if isinstance(exp, Iterable):
                    val_experiences_list.extend(exp.dataset)
            else:
                val_experiences_list.append(exp.dataset)
            val_experiences_list = AvalancheDataset(val_experiences_list)

            if self.strategy_name == "JointTraining":
                self.strategy.train(self.benchmark.train_stream, eval_streams=[self.benchmark.valid_stream])
                print('Training completed')

                if self.pp_calibration_mode:
                    self.model = ModelWithTemperature(self.model, self.device)
                    print("%%%% before calibrate temperature", self.model.temperature.data)
                    self.tb_logger.writer.add_scalar("temperature", self.model.temperature.data, 0)
                    self.calibrate_temperature(val_experiences_list)
                    optimal_temperature = self.model.temperature
                    print("%%%% after calibrate temperature", optimal_temperature.data)
                    self.tb_logger.writer.add_scalar("temperature", self.model.temperature.data, 1)

                print('Computing accuracy on the whole test set')
                # test also returns a dictionary which contains all the metric values
                results.append(self.strategy.eval(self.benchmark.test_stream))
            else:
                buffer_val = None
                for experience_tr, experience_val in zip(self.benchmark.train_stream, self.benchmark.valid_stream):
                    print("Start of experience: ", experience_tr.current_experience)
                    print("Current Classes: ", experience_tr.classes_in_this_experience)

                    # train returns a dictionary which contains all the metric values
                    self.strategy.train(experience_tr, eval_streams=[experience_val])
                    print('Training completed')

                    if self.pp_calibration_mode:
                        self.model = ModelWithTemperature(self.model, self.device)
                        print("%%%% before calibrate temperature", self.model.temperature.data)
                        self.tb_logger.writer.add_scalar("temperature", self.model.temperature.data, 0)

                        experience_val_data = make_classification_dataset(experience_val.dataset)
                        if buffer_val and self.pp_cal_mixed_data:
                            buffer_length = len(buffer_val)
                            indices = list(range(buffer_length))
                            np.random.shuffle(indices)
                            val_split_index = int(np.floor(0.4 * buffer_length))
                            new_buffer = AvalancheSubset(buffer_val, indices[:val_split_index])
                            buffer_val = AvalancheConcatDataset([new_buffer, experience_val_data])
                        else:
                            buffer_val = experience_val_data

                        print("!!!!!!! VAL Classes: !!!!!!!", experience_val.previous_classes, experience_val.classes_in_this_experience, len(buffer_val))
                        self.calibrate_temperature(buffer_val)
                        optimal_temperature = self.model.temperature
                        print("%%%% after calibrate temperature", optimal_temperature.data)
                        self.tb_logger.writer.add_scalar("temperature", self.model.temperature.data, 1)

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
        ece_metric = ECE()
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
        self.tb_logger.writer.add_scalar("NLL", before_temperature_nll, 0)
        self.tb_logger.writer.add_scalar("ECE", before_temperature_ece_metric, 0)
        ece_metric.reset()

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.model.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.model.temperature_scale(logits), labels).item()
        ece_metric.update(self.model.temperature_scale(logits), labels)
        after_temperature_ece_metric = ece_metric.result()
        print('##### Optimal temperature: %.3f' % self.model.temperature.data)
        print('##### After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece_metric))
        self.tb_logger.writer.add_scalar("NLL", after_temperature_nll, 1)
        self.tb_logger.writer.add_scalar("ECE", after_temperature_ece_metric, 1)
        ece_metric.reset()

        return self
