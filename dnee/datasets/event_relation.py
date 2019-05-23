import os
import json
import logging
import linecache

from torch.utils.data import ConcatDataset, Dataset
import torch

logger = logging.getLogger(__name__)


class EventRelationDataset(Dataset):
    def __init__(self, fpath, arg_lens):
        self.arg_lens = arg_lens
        logger.info("loading dataset: {}".format(fpath))
        logger.info('arg_lens={}'.format(arg_lens))
        self.fpath = fpath
        self._len = 0
        with open(self.fpath, 'r') as fr:
            self._len = len(fr.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self.fpath, idx+1)
        rel = json.loads(line)
        
        y = torch.zeros(1, dtype=torch.int64)
        y[0] = rel[0] if rel[0] == 1 else -1    # positive or negative
        
        x = self.raw2x(rel, self.arg_lens)
        return x, y

    def __len__(self):
        return self._len

    @staticmethod
    def raw2x(raw, arg_lens):
        e_len = 1 + arg_lens[0] + arg_lens[1]
        x = torch.zeros(2 * e_len + 1, dtype=torch.int64)
        x[0] = raw[1] # rtype
        e1_begin = 1
        x[e1_begin] = raw[2] # e1 predicate
        x[e1_begin+1: e1_begin+1+len(raw[3])] = torch.LongTensor(raw[3]) # e1 arg0
        x[e1_begin+1+arg_lens[0]: e1_begin+1+arg_lens[0]+len(raw[4])] = torch.LongTensor(raw[4]) # e1 arg1
        # rel[5] is e1 arg2
        e2_begin = e1_begin + 1 + arg_lens[0] + arg_lens[1]
        x[e2_begin] = raw[6] # e2 predicate
        x[e2_begin+1: e2_begin+1+len(raw[7])] = torch.LongTensor(raw[7]) # e2 arg0
        x[e2_begin+1+arg_lens[0]: e2_begin+1+arg_lens[0]+len(raw[8])] = torch.LongTensor(raw[8]) # e2 arg1
        # rel[9] is e2 arg2
        return x


class EventRelationConcatDataset(ConcatDataset):
    def __init__(self, fld, arg_lens):
        self.fpaths = [os.path.join(fld, f) for f in os.listdir(fld)]
        datasets = [EventRelationDataset(fp, arg_lens) for fp in self.fpaths]
        super(EventRelationConcatDataset, self).__init__(datasets)
