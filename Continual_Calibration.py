"""Continual calibration via temperature scaling
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
On Calibration of Modern Neural Networks.
Adapted from: https://github.com/gpleiss/temperature_scaling
"""
import copy
import torch as th
from torch import nn
from torch.nn import functional as F
from avalanche.training.supervised import Naive, JointTraining, Replay, DER
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset, AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_attribute import ConstantSequence
from ModelDecorator import ModelWithTemperature, MatrixAndVectorScaling
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
                 pp_cal_vector_scaling,
                 pp_cal_matrix_scaling,
                 calibration_mode_str,
                 num_classes,
                 lrpp,
                 max_iter,
                 num_bins,
                 batch_size_mem,
                 alpha,
                 beta,
                 logdir
                 ):
        self.lrpp = lrpp
        self.max_iter = max_iter
        self.tb_logger = tb_logger
        self.model = model
        self.num_classes = num_classes
        self.strategy_name = strategy_name
        self.benchmark = benchmark
        self.mem_size = mem_size
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.eval_mb_size = eval_mb_size
        self.batch_size_mem=batch_size_mem
        self.alpha=alpha
        self.beta=beta
        self.device = device
        self.eval_plugin = eval_plugin
        self.optimizer = optimizer
        self.criterion = criterion
        self.pp_calibration_mode = pp_calibration_mode
        self.pp_cal_mixed_data = pp_cal_mixed_data
        self.pp_cal_vector_scaling = pp_cal_vector_scaling
        self.pp_cal_matrix_scaling = pp_cal_matrix_scaling
        self.calibration_mode_str = calibration_mode_str
        self.num_bins = num_bins
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
            elif self.strategy_name == "DER":
                self.strategy = DER(
                    self.model,
                    self.optimizer,
                    mem_size=self.mem_size,
                    criterion=self.criterion,
                    train_mb_size=self.train_mb_size,
                    batch_size_mem=self.batch_size_mem,
                    alpha=self.alpha,
                    beta=self.beta,
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
                    # for parameters in self.strategy.model.parameters():
                    #     print(parameters.size())
                    #     print(parameters)
                    if self.pp_cal_vector_scaling:
                        self.strategy.model = MatrixAndVectorScaling(self.strategy.model, self.device, self.num_classes, self.num_bins, True)
                    elif self.pp_cal_matrix_scaling:
                        self.strategy.model = MatrixAndVectorScaling(self.strategy.model, self.device, self.num_classes, self.num_bins)
                    else:
                        self.strategy.model = ModelWithTemperature(self.strategy.model, self.device, self.num_bins)

                    self.strategy.model.calibrate(self.lrpp, self.max_iter, val_experiences_list)

                    # for parameters in self.strategy.model.parameters():
                    #     print(parameters.size())
                    #     print(parameters)
                print('Computing accuracy on the whole test set')
                # test also returns a dictionary which contains all the metric values
                results.append(self.strategy.eval(self.benchmark.test_stream))
                th.save(self.strategy.model.state_dict(), f"{self.log_dir}/model_{self.strategy_name}_{self.calibration_mode_str}.pt")
            else:
                buffer_val = None
                weights_pre_exp = None
                bias_pre_exp = None
                temperature_pre_exp = None
                for experience_tr, experience_val in zip(self.benchmark.train_stream, self.benchmark.valid_stream):
                    print("############### Start of experience: ", experience_tr.current_experience)
                    print("Current Classes: ", experience_tr.classes_in_this_experience)

                    # train returns a dictionary which contains all the metric values
                    self.strategy.train(experience_tr, eval_streams=[experience_val])
                    print('Training completed')

                    if self.pp_calibration_mode:
                        # if experience_tr.current_experience == 0:
                        if self.pp_cal_vector_scaling:
                            self.strategy.model = MatrixAndVectorScaling(self.strategy.model, self.device, self.num_classes, True)
                            if experience_tr.current_experience > 0:
                                self.strategy.model.weights_init(weights_pre_exp, bias_pre_exp)
                        elif self.pp_cal_matrix_scaling:
                            self.strategy.model = MatrixAndVectorScaling(self.strategy.model, self.device, self.num_classes, self.num_bins)
                            if experience_tr.current_experience > 0:
                                self.strategy.model.weights_init(weights_pre_exp, bias_pre_exp)
                        else:
                            self.strategy.model = ModelWithTemperature(self.strategy.model, self.device, self.num_bins)
                            if experience_tr.current_experience > 0:
                                self.strategy.model.temperature_init(temperature_pre_exp)

                        experience_val_data = make_classification_dataset(experience_val.dataset)
                        if buffer_val and self.pp_cal_mixed_data:
                            buffer_length = len(experience_val_data)
                            indices = list(range(buffer_length))
                            np.random.shuffle(indices)
                            val_split_index = int(np.floor(0.4 * buffer_length))
                            new_buffer = AvalancheSubset(experience_val_data, indices[:val_split_index])
                            buffer_val = AvalancheConcatDataset([new_buffer, buffer_val])
                        else:
                            buffer_val = experience_val_data

                        # print("!!!!!!! VAL Classes: !!!!!!!", experience_val.previous_classes, experience_val.classes_in_this_experience, len(buffer_val))
                        self.strategy.model.calibrate(self.lrpp, self.max_iter, buffer_val)

                    print('Computing accuracy on the whole test set')
                    # test also returns a dictionary which contains all the metric values
                    results.append(self.strategy.eval(self.benchmark.test_stream))

                    # store model after each experience
                    th.save(
                        self.strategy.model.state_dict(),
                        f"{self.log_dir}/model_{self.strategy_name}_{self.calibration_mode_str}_exp{experience_tr.current_experience}.pt"
                    )

                    if self.pp_calibration_mode:
                        if self.pp_cal_vector_scaling or self.pp_cal_matrix_scaling:
                            weights_pre_exp = self.strategy.model.weights
                            bias_pre_exp = self.strategy.model.bias
                        else:
                            temperature_pre_exp = self.strategy.model.temperature
                        self.strategy.model = copy.deepcopy(self.strategy.model.model)

            return results
