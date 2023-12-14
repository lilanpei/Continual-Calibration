from avalanche.training.supervised import Naive, JointTraining, Replay

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
                 device
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
                for experience in self.benchmark.train_stream:
                    print("Start of experience: ", experience.current_experience)
                    print("Current Classes: ", experience.classes_in_this_experience)

                    # train returns a dictionary which contains all the metric values
                    self.strategy.train(experience)
                    print('Training completed')

                    print('Computing accuracy on the whole test set')
                    # test also returns a dictionary which contains all the metric values
                    results.append(self.strategy.eval(self.benchmark.test_stream))
            return results
