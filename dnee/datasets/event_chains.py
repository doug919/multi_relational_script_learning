import os
import json
import logging
import linecache

from torch.utils.data import Dataset
import torch


class EventChainDataset(Dataset):
    def __init__(self, fpath, n_args=3, n_pos=9, n_neg=4):
        super(EventChainDataset, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("loading dataset: {}".format(fpath))
        self.logger.info('arg_lens={}'.format(arg_lens))
        self.n_args = n_args
        self.n_pos, self.n_neg = n_pos, n_neg
        self.fpath = fpath
        self._len = 0
        with open(self.fpath, 'r') as fr:
            self._len = len(fr.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self.fpath, idx+1)
        indexed_events = json.loads(line)
        
        x = torch.zeros((1+n_args)*(n_pos+n_neg), dtype=torch.int64)
        for i, ie in enumerate(indexed_events):
            start = i * (1+n_args)
            end = (i+1) * (1+n_args)
            import pdb; pdb.set_trace()
            x[start:end] = ie
        return x

    def __len__(self):
        return self._len
