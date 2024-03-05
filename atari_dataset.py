import torch as th
import numpy as np
import gzip

from torch.utils.data import Dataset, random_split
from avalanche.benchmarks.generators import dataset_benchmark

class AtariDataset(Dataset):
    __attributes = ['observation', 'action', 'terminal']
    __store_prefix = '$store$_'
    __dataset_max_size = 1000000

    def __init__(self, game, data_idx, ckp_idx, size=__dataset_max_size, stack_size=4):
        self.game = game
        self.data_idx = data_idx
        self.ckp_idx = ckp_idx
        self.size = size
        assert(size <= self.__dataset_max_size)
        self.stack_size = stack_size

        self.data = self.__load_data()
        # print("Loaded data - checking elements")

        self.actual_size = 0
        self.obs = []
        self.targets = []
        for i in range(size):
            if self.__check_valid_index(i):
                self.actual_size += 1
                self.obs.append(self.__get_stack('observation', i))
                self.targets.append(self.data['action'][i].long())

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        return self.obs[index], self.targets[index]
    
    def __load_data(self):
        data_dir = f'datasets/{self.game}/{self.data_idx}/replay_logs/'
        print("Loading data:", data_dir)
        if self.size < self.__dataset_max_size:
            data_idxs = np.random.randint(0, self.__dataset_max_size, size=self.size)
        else:
            data_idxs = range(self.__dataset_max_size)
        # print("data_idxs:", data_idxs)
        data = {}
        for attr in self.__attributes:
            filename = f'{data_dir}{self.__store_prefix}{attr}_ckpt.{self.ckp_idx}.gz'
            with open(filename, 'rb') as f:
                with gzip.GzipFile(fileobj=f) as infile:
                    tmp = np.load(infile)
                    data[attr] = th.from_numpy(tmp[data_idxs, ...])
        return data

    def __check_valid_index(self, idx):
        if idx < self.stack_size - 1:
            return False
        
        # check no terminal state in other frames
        terminal_stack = self.__get_stack('terminal', idx)
        if terminal_stack[:-1].any():
            return False

        return True

    # [index - state_stack + 1 : index + 1]
    def __get_stack(self, attr, idx):
        return self.data[attr][idx - self.stack_size + 1: idx + 1, ...]


def generate_atari_benchmark(n_experinces, data_idx=1, ckp_idx=49, dataset_size=200000, seed=1):
    # default ckp_idx 49 - last one from training (https://github.com/google-research/batch_rl/issues/33)

    # sorted wrt performance in https://daiwk.github.io/assets/dqn.pdf
    GAMES_AVAILABLE = ['VideoPinball', 'Boxing', 'Breakout', 'StarGunner', 'Atlantis']
    
    assert(n_experinces <= len(GAMES_AVAILABLE))
    assert(data_idx < 6)
    assert(ckp_idx < 50)

    train_dt = []
    test_dt = []
    for i in range(n_experinces):
        dataset = AtariDataset(GAMES_AVAILABLE[i], data_idx, ckp_idx, dataset_size)
        train, test = random_split(dataset, [0.7, 0.3], th.Generator().manual_seed(seed))
        train_dt.append(train)
        test_dt.append(test)
    
    return dataset_benchmark(train_datasets=train_dt, test_datasets=test_dt)


if __name__ == "__main__":
    dataset = AtariDataset('Atlantis', 1, 50, 10)

    train, test = random_split(dataset, [0.7, 0.3], th.Generator().manual_seed(1))

    for obs, action in dataset:
        print("obs:", obs, obs.shape, type(obs))
        print("action:", action, type(action))
        print("--------")

    for obs, action in train:
        print("obs:", obs, obs.shape, type(obs))
        print("action:", action, type(action))
        print("--------")

    print("dataset_size:", len(dataset), type(dataset))
    print("train_dataset_size:", len(train), type(train))
    print("test_dataset_size:", len(test), type(test))