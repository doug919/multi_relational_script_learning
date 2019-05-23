import os
import linecache
import logging

import numpy as np
import torch
from torch.utils.data import Dataset



class EventCompDataset(Dataset):
    def __init__(self, fpath, use_torch=True):
        super(EventCompDataset, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fpath = fpath
        self.use_torch = use_torch
        with open(self.fpath, 'r') as fr:
            self._len = len(fr.readlines()) - 1
        
    def __getitem__(self, idx):
        line = linecache.getline(self.fpath, idx+1)
        x = np.array([int(n) for n in line.split(' ')], dtype=np.int64)
        if self.use_torch:
            x = torch.from_numpy(x)
        # e1 [0:4], e2 [4:8] ne [8:12]
        return x
    
    def __len__(self):
        return self._len
