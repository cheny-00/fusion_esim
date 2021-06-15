import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class DistillationDataset(Dataset):

    def __init__(self,
                 data_path,):
        super(DistillationDataset, self).__init__()

        self.data_path = data_path
        f = open(data_path, 'rb')
        self.data = pickle.load(f)


    def __len__(self):
        return len(self.data)
