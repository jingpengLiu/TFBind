import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import nuc_utils

class MyDataSet(Dataset):
    def __init__(self, nuc_data, nuc_label, data_type):
        self.length = nuc_data.__len__()
        train_location = int(self.length * 0.8)
        self.y = torch.tensor(nuc_label)
        if data_type == "train":
            self.x = nuc_data[0:train_location]
            self.y = nuc_label[0:train_location]
        elif data_type == "val":
            self.x = nuc_data[train_location:]
            self.y = nuc_label[train_location:]
        elif data_type == "test":
            self.x = nuc_data
            self.y = nuc_label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]




if __name__ == '__main__':
    print("Hello")



