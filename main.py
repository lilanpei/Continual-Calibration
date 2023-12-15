import argparse
import torch as th
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy
from Continual_Calibration import Continual_Calibration
from ECE_metrics import ExperienceECE
from Ent_Loss import Ent_Loss

if __name__ == "__main__":
    validation_size = 0.2
    foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
    benchmark = SplitMNIST(n_experiences=5)
    bm = benchmark_with_validation_stream(benchmark, custom_split_strategy=foo)
    model = SimpleMLP(num_classes=benchmark.n_classes)
    mem_size = 300
    train_mb_size=32
    train_epochs=2
    eval_mb_size=32
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # criterion = CrossEntropyLoss()
    ent_weight = 1e-3
    criterion = Ent_Loss(ent_weight)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    strategy_name = "Naive"
    
    # log to Tensorboard
    tb_logger = TensorboardLogger()
    # log to text file
    text_logger = TextLogger(open('log.txt', 'a'))
    # print to stdout
    interactive_logger = InteractiveLogger()
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceECE(),  # after training on each experience it computes ECE on each experience
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    
    continual_calibration = Continual_Calibration(model, optimizer, criterion, strategy_name, bm, train_mb_size, train_epochs, mem_size, eval_mb_size, eval_plugin, device)
    continual_calibration.train()