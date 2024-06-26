# import os
import ssl
import argparse
import torch as th
import pickle
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from torchvision import transforms, models
from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR100, SplitTinyImageNet
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP, pytorchcv_wrapper
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, LwFPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy
from Continual_Calibration import Continual_Calibration
from ECE_metrics import ExperienceECE, ExpECEHistogram
from Ent_Loss import Ent_Loss
from atari_dataset import generate_atari_benchmark
from DQN_model import DQNModel
from ResNet18 import resnet18

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts",
        "--train_mb_size",
        type=int,
        default=32,
        help="mini batch size for training",
    )
    parser.add_argument(
        "-es",
        "--eval_mb_size",
        type=int,
        default=32,
        help="mini batch size for evaluation",
    )
    parser.add_argument(
        "-tp",
        "--train_epochs",
        type=int,
        default=2,
        help="number of epochs for training",
    )
    parser.add_argument(
        "-ms",
        "--mem_size",
        type=int,
        default=300,
        help="replay buffer size",
    )
    parser.add_argument(
        "-vs",
        "--validation_split",
        type=float,
        default=0.2,
        help="validation split size",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "-lrpp",
        "--learning_rate_for_ppcm",
        type=float,
        default=0.01,
        help="learning rate for post processing calibration",
    )
    parser.add_argument(
        "-mi",
        "--max_iter",
        type=float,
        default=50,
        help="max iteration for post processing calibration",
    )
    parser.add_argument(
        "-t0",
        "--T0",
        type=int,
        default=3,
        help="Number of iterations for the first restart of CosineAnnealingWarmRestarts lr_scheduler",
    )
    parser.add_argument(
        "-ew",
        "--ent_weight",
        type=float,
        default=1e-3,
        help="entropy weight",
    )
    parser.add_argument(
        "-sn",
        "--strategy_name",
        type=str,
        default="Naive",
        help="strategy name",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="SplitMNIST",
        help="dataset name",
    )
    parser.add_argument(
        "-stcm",
        "--self_training_calibration_mode",
        help="self training calibration mode",
        action="store_true",
    )
    parser.add_argument(
        "-ppcm",
        "--post_processing_calibration_mode",
        help="post processing calibration mode",
        action="store_true",
    )
    parser.add_argument(
        "-ppdm",
        "--post_processing_calibration_mixed_data",
        help="post processing calibration with mixed data",
        action="store_true",
    )
    parser.add_argument(
        "-ppvs",
        "--post_processing_calibration_vector_scaling",
        help="post processing calibration with vector scaling",
        action="store_true",
    )
    parser.add_argument(
        "-ppms",
        "--post_processing_calibration_matrix_scaling",
        help="post processing calibration with matrix scaling",
        action="store_true",
    )
    parser.add_argument(
        "-ld",
        "--logdir",
        type=str,
        help="logging directory",
    )
    parser.add_argument(
        "-cid",
        "--cuda_id",
        type=str,
        default="0",
        help="cuda gpu index",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait without generalization",
    )
    parser.add_argument(
        "-nb",
        "--num_bins",
        type=int,
        default=10,
        help="Number of bins in ECE Histogram",
    )
    parser.add_argument(
        "-ep",
        "--early_stopping",
        help="Early stopping",
        action="store_true",
    )
    parser.add_argument(
        "-lwf",
        "--LearningWithoutForgetting",
        help="Learning Without Forgetting method applies knowledge distilllation to mitigate forgetting",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="1",
        help="run version",
    )
    parser.add_argument(
        "-bsm",
        "--batch_size_mem",
        help="Size of the batch sampled from the DER buffer",
        default=None
    )
    parser.add_argument(
        "-a",
        "--alpha",
        help="DER hyperparameter weighting the MSE loss",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "-b",
        "--beta",
        help="DER hyperparameter weighting the CE loss",
        type=float,
        default=0.5
    )

    args = parser.parse_args()
    th.set_num_threads(1)
    plugins = []

    if args.batch_size_mem:
        batch_size_mem = int(args.batch_size_mem)
    else:
        batch_size_mem = None

    if args.dataset_name == "SplitCIFAR100":
        benchmark = SplitCIFAR100(n_experiences=10)
        model = pytorchcv_wrapper.resnet("cifar100", depth=110, pretrained=False)
        model_name = "ResNet110"
        num_classes = 100
        milestones=[60, 120, 160]
    elif args.dataset_name == "EuroSAT":
        # --- TRANSFORMATIONS
        transform = transforms.Compose([ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        # --- BENCHMARK CREATION
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = EuroSAT(root=".", transform=transform, download=True)
        n = int(len(dataset) * 0.9)
        eurosat_train, eurosat_test = th.utils.data.random_split(dataset, [n, len(dataset) - n])
        benchmark = nc_benchmark(
            eurosat_train,
            eurosat_test,
            5,
            task_labels=False
        )
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = 3, bias = False)
        model_name = "ResNet50"
        num_classes = 10
        milestones = [50,75,90]
    elif args.dataset_name == "Atari":
        benchmark = generate_atari_benchmark(n_experinces=5)
        model = DQNModel(num_actions=18)
        model_name = "NatureDQNNetwork"
        num_classes = 18
        milestones = None
    elif args.dataset_name == "TinyImageNet":
        benchmark = SplitTinyImageNet(n_experiences=10)
        num_classes = 200
        model = resnet18(num_classes)
        model_name = "ResNet18"
        milestones = None
    else:
        benchmark = SplitMNIST(n_experiences=5)
        model = SimpleMLP(num_classes=benchmark.n_classes)
        model_name = "SimpleMLP"
        num_classes = 10
        milestones = None

    foo = lambda exp: class_balanced_split_strategy(args.validation_split, exp)
    bm = benchmark_with_validation_stream(benchmark, custom_split_strategy=foo)
    mem_size = args.mem_size
    train_mb_size = args.train_mb_size
    train_epochs = args.train_epochs
    eval_mb_size = args.eval_mb_size
    
    if args.dataset_name == "Atari":
         optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.dataset_name in ["SplitCIFAR100", "EuroSAT"]:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=0, momentum=0)
    
    if milestones:
        if args.dataset_name in ["SplitCIFAR100", "EuroSAT"]:
            sched = LRSchedulerPlugin(CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=1, eta_min=1e-5))
        else:
            sched = LRSchedulerPlugin(
                    MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay
                )
        plugins.append(sched)

    ent_weight = args.ent_weight
    if args.early_stopping:
        early_stopping = EarlyStoppingPlugin(patience=args.patience, val_stream_name='valid_stream')
        plugins.append(early_stopping)

    if args.LearningWithoutForgetting:
        lwf = LwFPlugin()
        plugins.append(lwf)

    if args.early_stopping:
        early_stopping = EarlyStoppingPlugin(patience=args.patience, val_stream_name='valid_stream')
        plugins.append(early_stopping)

    if args.self_training_calibration_mode:
        criterion = Ent_Loss(ent_weight)
        calibration_mode = "SelfTraining_" + str(ent_weight)
    else:
        criterion = CrossEntropyLoss()
        calibration_mode = "NoSelfTraining"

    device = th.device(f"cuda:{args.cuda_id}" if th.cuda.is_available() else "cpu")
    strategy_name = args.strategy_name
    pp_calibration_mode = args.post_processing_calibration_mode
    pp_cal_mixed_data = args.post_processing_calibration_mixed_data
    pp_cal_vector_scaling = args.post_processing_calibration_vector_scaling
    pp_cal_matrix_scaling = args.post_processing_calibration_matrix_scaling

    if pp_calibration_mode:
        calibration_mode = calibration_mode + "_" + "PostProcessing"

        if pp_cal_vector_scaling:
            calibration_mode = calibration_mode + "_VectorScaling"
        elif pp_cal_matrix_scaling:
            calibration_mode = calibration_mode + "_MatrixScaling"
        else:
            calibration_mode = calibration_mode + "_TemperatureScaling"

        if pp_cal_mixed_data:
            calibration_mode = calibration_mode + "_MixedData"
    else:
        calibration_mode = calibration_mode + "_" + "NoPostProcessing"

    calibration_mode += args.version

    # log to Tensorboard
    tb_logger = TensorboardLogger(f'{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}')
    # log to text file
    text_logger = TextLogger(open(f'{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}_log.txt', 'a'))
    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceECE(num_bins=args.num_bins),  # after training on each experience it computes ECE on each experience
        ExpECEHistogram(num_bins=args.num_bins),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    continual_calibration = Continual_Calibration(tb_logger, model, optimizer, plugins, criterion, strategy_name, bm, train_mb_size, train_epochs, mem_size, eval_mb_size, eval_plugin, device, pp_calibration_mode, pp_cal_mixed_data, pp_cal_vector_scaling, pp_cal_matrix_scaling, calibration_mode, num_classes, args.learning_rate_for_ppcm, args.max_iter, args.num_bins, batch_size_mem, args.alpha, args.beta, args.logdir)
    res = continual_calibration.train()

    with open(f"{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}_dict", "wb") as file:
        pickle.dump(res, file)
