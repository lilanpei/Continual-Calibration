import torch as th
import numpy as np
import gzip

from torch.utils.data import Dataset

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
        print("Loaded data - checking elements")

        self.actual_size = 0
        self.obs = []
        self.actions = []
        for i in range(size):
            if self.__check_valid_index(i):
                self.actual_size += 1
                self.obs.append(self.__get_stack('observation', i))
                self.actions.append(self.data['action'][i])

    def __len__(self):
        return self.actual_size
    
    def __getitem__(self, index):
        return self.obs[index], self.actions[index]
    
    def __load_data(self):
        data_dir = f'datasets/{self.game}/{self.data_idx}/replay_logs/'
        print("Loading data:", data_dir)
        if self.size < self.__dataset_max_size:
            data_idxs = np.random.randint(0, self.__dataset_max_size, size=self.size)
        else:
            data_idxs = range(self.__dataset_max_size)
        print("data_idxs:", data_idxs)
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


if __name__ == "__main__":
    dataset = AtariDataset('Atlantis', 1, 50, 10)

    for obs, action in dataset:
        print("obs:", obs, obs.shape)
        print("action:", action)
        print("--------")
    print("dataset_size:", len(dataset))