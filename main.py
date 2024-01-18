import os
import ssl
import argparse
import torch as th
import pickle
from torch.optim import SGD, Adam
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, models
from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP, pytorchcv_wrapper
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy
from Continual_Calibration import Continual_Calibration
from ECE_metrics import ExperienceECE, ExpECEHistogram
from Ent_Loss import Ent_Loss

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
        "-m",
        "--momentum",
        type=float,
        default=0.9,
        help="momentum",
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
        help="Number of epochs to wait without generalization"
        "improvements before stopping the training .",
    )

    parser.add_argument(
        "-ep",
        "--early_stopping",
        help="Early stopping",
        action="store_true",
    )

    args = parser.parse_args()
    th.set_num_threads(1)
    th.random.manual_seed(42)
    plugins = []
    milestones = None

    if args.dataset_name == "SplitCIFAR100":
        benchmark = SplitCIFAR100(n_experiences=10)
        model = pytorchcv_wrapper.resnet("cifar100", depth=110, pretrained=False)
        milestones=[60, 120, 160]
        model_name = "ResNet110"
    elif args.dataset_name == "EuroSAT":
        # --- TRANSFORMATIONS
        transform = transforms.Compose([ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        # --- BENCHMARK CREATION
        # root = os.path.join(root, SUBDIR)
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = EuroSAT(root=".", transform=transform, download=True)
        n = int(len(dataset) * 0.9)
        eurosat_train, eurosat_test = th.utils.data.random_split(dataset, [n, len(dataset) - n])
        benchmark = nc_benchmark(
            eurosat_train,
            eurosat_test,
            5,
            task_labels=False,
            seed=1234,
            fixed_class_order=[i for i in range(10)],
        )
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = 3, bias = False)
        milestones = [50,75,90]
        model_name = "ResNet50"
    else:
        benchmark = SplitMNIST(n_experiences=5)
        model = SimpleMLP(num_classes=benchmark.n_classes)
        model_name = "SimpleMLP"

    foo = lambda exp: class_balanced_split_strategy(args.validation_split, exp)
    bm = benchmark_with_validation_stream(benchmark, custom_split_strategy=foo)
    mem_size = args.mem_size
    train_mb_size = args.train_mb_size
    train_epochs = args.train_epochs
    eval_mb_size = args.eval_mb_size
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # if milestones:
    #     sched = LRSchedulerPlugin(
    #                 MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay
    #             )
    #     plugins.append(sched)
    ent_weight = args.ent_weight
    if args.early_stopping:
        early_stopping = EarlyStoppingPlugin(patience=args.patience, val_stream_name='valid_stream')
        plugins.append(early_stopping)

    if args.self_training_calibration_mode:
        criterion = Ent_Loss(ent_weight)
        calibration_mode = "SelfTraining"
    else:
        criterion = CrossEntropyLoss()
        calibration_mode = "NoSelfTraining"
    
    device = th.device(f"cuda:{args.cuda_id}" if th.cuda.is_available() else "cpu")
    strategy_name = args.strategy_name
    pp_calibration_mode = args.post_processing_calibration_mode
    pp_cal_mixed_data = args.post_processing_calibration_mixed_data

    if pp_calibration_mode:
        calibration_mode = calibration_mode + "_" + "PostProcessing"
        if pp_cal_mixed_data:
            calibration_mode = calibration_mode + "_MixedData"
    else:
        calibration_mode = calibration_mode + "_" + "NoPostProcessing"

    # log to Tensorboard
    tb_logger = TensorboardLogger(f'{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}')
    # log to text file
    text_logger = TextLogger(open(f'{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}_log.txt', 'a'))
    # print to stdout
    interactive_logger = InteractiveLogger()
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceECE(),  # after training on each experience it computes ECE on each experience
        ExpECEHistogram(),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    continual_calibration = Continual_Calibration(tb_logger, model, optimizer, plugins, criterion, strategy_name, bm, train_mb_size, train_epochs, mem_size, eval_mb_size, eval_plugin, device, pp_calibration_mode, pp_cal_mixed_data, calibration_mode, args.logdir)
    res = continual_calibration.train()

    with open(f"{args.logdir}/{args.dataset_name}_{model_name}_{strategy_name}_{calibration_mode}_dict", "wb") as file:
        pickle.dump(res, file)