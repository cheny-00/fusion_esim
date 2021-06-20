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
        return len(self.data['label'])


    @staticmethod
    def formulate_dataset(idx, data):
        features = dict()
        context, response, c_len, r_len, label, t_logits = data['context'][idx], \
                                                           data['response'][idx],\
                                                           data['c_len'][idx],   \
                                                           data['r_len'][idx],   \
                                                           data['label'][idx],   \
                                                           data['t_logits'][idx]
        features['esim_data'] = ((torch.tensor(context, dtype=torch.long), torch.tensor(c_len, dtype=torch.long)),
                                 (torch.tensor(response, dtype=torch.long), torch.tensor(r_len, dtype=torch.long)))
        features['t_logits'] = torch.tensor(t_logits, dtype=torch.float)
        features['label'] = torch.tensor(label, dtype=torch.long)

        return features


    def __getitem__(self, idx):

        return self.formulate_dataset(idx, self.data)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    path = '../checkpoints/create_distillation_dataset/20210615-104832/distillation_dataset.pkl'
    dataset = DistillationDataset(path)
    data_iter = DataLoader(dataset, batch_size=16, num_workers=8)