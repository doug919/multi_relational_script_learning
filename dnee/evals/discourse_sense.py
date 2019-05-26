import os
import re
import json
import logging
from collections import OrderedDict

import h5py
import six
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from allennlp.modules.elmo import batch_to_ids
from nltk import word_tokenize

from ..datasets.event_relation import EventRelationDataset
from ..events import Event, EventRelation
from .confusion_matrix import Alphabet, ConfusionMatrix


DNEE_SENSE_MAP_9REL = {'Temporal.Asynchronous.Precedence': 'Temporal.Asynchronous',
                'Temporal.Asynchronous.Succession': 'Temporal.Asynchronous',
                'Temporal.Synchrony': 'Temporal.Synchrony',
                'Contingency.Cause.Reason': 'Contingency.Cause.Reason',
                'Contingency.Cause.Result': 'Contingency.Cause.Result',
                'Contingency.Condition': 'Contingency.Condition',
                'Comparison.Contrast': 'Comparison.Contrast',
                'Expansion.Conjunction': 'Expansion.Conjunction',
                'Expansion.Instantiation': 'Expansion.Instantiation',
                'Expansion.Restatement': 'Expansion.Restatement'
                }

DNEE_SENSE_MAP_4REL = {'Temporal.Asynchronous.Precedence': 'Temporal',
        'Temporal.Asynchronous.Succession': 'Temporal',
        'Temporal.Synchrony': 'Temporal',
        'Contingency.Cause.Reason': 'Contingency',
        'Contingency.Cause.Result': 'Contingency',
        'Contingency.Condition': 'Contingency',
        'Comparison.Contrast': 'Comparison',
        'Comparison.Concession': 'Comparison',
        'Expansion.Conjunction': 'Expansion',
        'Expansion.Instantiation': 'Expansion',
        'Expansion.Restatement': 'Expansion',
        'Expansion.Alternative': 'Expansion',
        'Expansion.Alternative.Chosen alternative': 'Expansion',
        'Expansion.Exception': 'Expansion'
        }


class AttentionNN(torch.nn.Module):
    def __init__(self, n_rel_classes, arg_dim=512, dnee_score_dim=15, event_dim=None, dropout=0.0,
            use_event=False, use_dnee_scores=True):
        super(AttentionNN, self).__init__()
        self.dropout = dropout
        self.arg_dim = arg_dim
        self.n_rel_classes = n_rel_classes
        self.event_dim = event_dim
        self.dnee_score_dim = dnee_score_dim
        self.use_event = use_event
        self.use_dnee_scores = use_dnee_scores

        self.arg_attn = torch.nn.Parameter(torch.FloatTensor(2, arg_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.arg_attn.data)
        self.l1_0 = torch.nn.Linear(arg_dim, arg_dim//2)
        self.l1_1 = torch.nn.Linear(arg_dim, arg_dim//2)
        if use_event:
            self.event_attn = torch.nn.Parameter(torch.FloatTensor(2, event_dim), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.event_attn.data)
            self.l1_e0 = torch.nn.Linear(event_dim, event_dim//2)
            self.l1_e1 = torch.nn.Linear(event_dim, event_dim//2)
            
            self.l1_s = torch.nn.Linear(dnee_score_dim, dnee_score_dim)
        
            torch.nn.init.xavier_uniform_(self.l1_e0.weight.data)
            torch.nn.init.xavier_uniform_(self.l1_e1.weight.data)
            self.l1_e0.bias.data.zero_()
            self.l1_e1.bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.l1_s.weight.data)
            self.l1_s.bias.data.zero_()
            dim = arg_dim + event_dim
            if self.use_dnee_scores:
                dim += dnee_score_dim
        else:
            dim = arg_dim
        self.d1 = torch.nn.Dropout(p=self.dropout)
        
        self.l2= torch.nn.Linear(dim, dim//2)
        self.d2 = torch.nn.Dropout(p=self.dropout)

        self.l3= torch.nn.Linear(dim//2, dim//4)
        self.d3 = torch.nn.Dropout(p=self.dropout)
        
        self.l4= torch.nn.Linear(dim//4, n_rel_classes)
        
        torch.nn.init.xavier_uniform_(self.l1_0.weight.data)
        torch.nn.init.xavier_uniform_(self.l1_1.weight.data)
        torch.nn.init.xavier_uniform_(self.l2.weight.data)
        torch.nn.init.xavier_uniform_(self.l3.weight.data)
        torch.nn.init.xavier_uniform_(self.l4.weight.data)
        self.l1_0.bias.data.zero_()
        self.l1_1.bias.data.zero_()
        self.l2.bias.data.zero_()
        self.l3.bias.data.zero_()
        self.l4.bias.data.zero_()

        self._loss = torch.nn.NLLLoss()

    @staticmethod
    def attend_context(x, attn):
        attention_score = torch.matmul(x, attn).squeeze()
        attention_score = F.softmax(attention_score, dim=1).view(x.size(0), x.size(1), 1)
        scored_x = x * attention_score
        x_context = torch.sum(scored_x, dim=1)
        return x_context

    def forward(self, x0, x1, x0_dnee, x1_dnee, x_dnee):
        x0_context = self.attend_context(x0, self.arg_attn[0])
        x1_context = self.attend_context(x1, self.arg_attn[1])
        out1_0 = F.relu(self.l1_0(x0_context))
        out1_1 = F.relu(self.l1_1(x1_context))
    
        if self.use_event:
            e0_context = self.attend_context(x0_dnee, self.event_attn[0])
            e1_context = self.attend_context(x1_dnee, self.event_attn[1])

            out1_e0 = F.relu(self.l1_e0(e0_context))
            out1_e1 = F.relu(self.l1_e1(e1_context))
            if self.use_dnee_scores:
                x_dnee = F.normalize(x_dnee)
                out1_s = F.relu(self.l1_s(x_dnee))
                out1 = torch.cat((out1_0, out1_1, out1_e0, out1_e1, out1_s), dim=1)
            else:
                out1 = torch.cat((out1_0, out1_1, out1_e0, out1_e1), dim=1)
        else:
            out1 = torch.cat((out1_0, out1_1), dim=1)
        out1 = self.d1(out1)
        out2 = self.d2(F.relu(self.l2(out1)))
        out3 = self.d3(F.relu(self.l3(out2)))
        out4 = F.log_softmax(self.l4(out3), dim=1)
        return out4

    def loss_func(self, probs, target):
        return self._loss(probs, target)

    def predict(self, x0, x1, x0_dnee, x1_dnee, x_dnee):
        with torch.no_grad():
            probs = self.forward(x0, x1, x0_dnee, x1_dnee, x_dnee)
            _, pred = torch.max(probs, 1)
        return pred


class DsDataset(Dataset):
    def __init__(self, fld, dnee_fld=None):
        super(DsDataset, self).__init__()
        self.fld = fld
        fpath = os.path.join(fld, 'data.h5')
        self.h5_file = h5py.File(fpath)
        self.x0 = self.h5_file.get('x0')
        self.x1 = self.h5_file.get('x1')
        self.y = self.h5_file.get('y')

        self._len = self.y.shape[0]
        assert self.x0.shape[0] == self._len
        assert self.x1.shape[0] == self._len
        self.seq_len = self.x0[0].shape[0]
        self.dnee_fld = None
        if dnee_fld:
            self.dnee_fld = dnee_fld
            fpath = os.path.join(dnee_fld, 'data.h5')
            self.h5_dnee = h5py.File(fpath)
            self.x0_dnee = self.h5_dnee.get('x0_dnee')
            self.x1_dnee = self.h5_dnee.get('x1_dnee')
            assert self.x0_dnee.shape == self.x1_dnee.shape
            self.dnee_seq_len = self.x0_dnee[0].shape[0]
            self.x_dnee_score = self.h5_dnee.get('x_dnee_score')
            assert self.x0_dnee.shape[0] == self._len
            assert self.x_dnee_score.shape[0] == self._len

    def __getitem__(self, idx):
        x0 = torch.from_numpy(self.x0[idx])
        x1 = torch.from_numpy(self.x1[idx])
        y = torch.LongTensor([self.y[idx]])

        if self.dnee_fld:
            x0_dnee = self.x0_dnee[idx]
            x1_dnee = self.x1_dnee[idx]
            x_dnee = self.x_dnee_score[idx]
            return (x0, x1, x0_dnee, x1_dnee, x_dnee), y
        return (x0, x1), y

    def __len__(self):
        return self._len


def get_features(rels, elmo, elmo_seq_len, model, event_seq_len, config, pred2idx, argw2idx, rel2idx, device=None, use_dnee=False):
    x0_elmo, x1_elmo, y_elmo = elmo_features(elmo, rels, rel2idx, seq_len=elmo_seq_len)
    x0_dnee, x1_dnee, x_dnee_score = None, None, None
    if use_dnee:
        x_dnee_score, _ = dnee_score_features(rels, model, config, pred2idx, argw2idx,
                                                rel2idx, device)
        x0_dnee, x1_dnee, y_dnee = dnee_ee_features(rels, model, config, pred2idx, argw2idx, event_seq_len,
                                                rel2idx, device)
    return (x0_elmo, x1_elmo, x0_dnee, x1_dnee, x_dnee_score), y_elmo


def elmo_features(elmo, rels, rel2idx, seq_len=None):
    arg0s = [word_tokenize(rel['Arg1']['RawText']) for rel in rels]
    char_ids = batch_to_ids(arg0s)
    a0we = elmo(char_ids)['elmo_representations'][0]
    
    arg1s = [word_tokenize(rel['Arg2']['RawText']) for rel in rels]
    char_ids = batch_to_ids(arg1s)
    a1we = elmo(char_ids)['elmo_representations'][0]
    
    ys = torch.LongTensor([rel2idx[rel['Sense'][0]] for rel in rels])
    
    if seq_len: 
        if a0we.shape[1] > seq_len:
            a0we = a0we[:, :seq_len, :]
        elif a0we.shape[1] < seq_len:
            dims = (a0we.shape[0], seq_len, a0we.shape[2])
            tmp = torch.zeros(dims, dtype=torch.float32)
            tmp[:, :a0we.shape[1]] = a0we
            a0we = tmp
        if a1we.shape[1] > seq_len:
            a1we = a1we[:, :seq_len, :]
        elif a1we.shape[1] < seq_len:
            dims = (a1we.shape[0], seq_len, a1we.shape[2])
            tmp = torch.zeros(dims, dtype=torch.float32)
            tmp[:, :a1we.shape[1]] = a1we
            a1we = tmp
    return a0we, a1we, ys


def we_comb_func(toks, we):
    dim = we[we.keys()[0]].shape[0]
    total_emb = torch.zeros(dim, dtype=torch.float32)
    cnt = 0

    for t in toks:
        if t in we:
            total_emb += we[t]
            cnt += 1

    if cnt > 0:
        total_emb = total_emb / cnt
    return total_emb


def we_features(rels, we, comb_func, device):
    we0s, we1s = [], []
    for i_rel, rel in enumerate(rels):
        a0 = clean_and_tokenize(rel['Arg1']['RawText'])
        a1 = clean_and_tokenize(rel['Arg2']['RawText'])

        we0 = comb_func(a0, we)
        we0s.append(we0)
        we1 = comb_func(a1, we)
        we1s.append(we1)
    f0 = torch.stack(we0s, dim=0).to(device)
    f1 = torch.stack(we1s, dim=0).to(device)
    return f0, f1


def st_features(rels, model, config, argw2idx, unknown_word, device):
    a0s, a1s = [], []
    max_len, max_len = 0, 0
    for i_rel, rel in enumerate(rels):
        a0 = encode_argument(rel['Arg1']['RawText'].lower(), argw2idx, unknown_word)
        if len(a0) > max_len:
            max_len = len(a0)
        a0s.append(a0)
        a1 = encode_argument(rel['Arg2']['RawText'].lower(), argw2idx, unknown_word)
        if len(a1) > max_len:
            max_len = len(a1)
        a1s.append(a1)

    ta0s, ta1s = [], []
    for a0, a1 in zip(a0s, a1s):
        t0, t1 = torch.zeros(max_len, dtype=torch.int64), \
                torch.zeros(max_len, dtype=torch.int64)
        t0[:len(a0)] = torch.LongTensor(a0)
        t1[:len(a1)] = torch.LongTensor(a1)
        ta0s.append(t0)
        ta1s.append(t1)
    a0s = torch.stack(ta0s, dim=0).to(device)
    a1s = torch.stack(ta1s, dim=0).to(device)
    with torch.no_grad():
        f0 = model(a0s)
        f1 = model(a1s)
    return f0, f1


def get_raw_event_repr(e, config, pred2idx, argw2idx, device=None, use_head=False):
    e_len = 1 + config['arg0_max_len'] + config['arg1_max_len']
    raw = torch.zeros(e_len, dtype=torch.int64).to(device) if device else torch.zeros(e_len, dtype=torch.int64)
    pred_idx = e.get_pred_index(pred2idx)
    arg0_idxs = e.get_arg_indices(0, argw2idx, arg_len=config['arg0_max_len'], use_head=use_head)
    arg1_idxs = e.get_arg_indices(1, argw2idx, arg_len=config['arg1_max_len'], use_head=use_head)

    raw[0] = pred_idx
    raw[1: 1+len(arg0_idxs)] = torch.LongTensor(arg0_idxs).to(device) if device else torch.LongTensor(arg0_idxs)
    raw[1+config['arg0_max_len']: 1+config['arg0_max_len']+len(arg1_idxs)] = torch.LongTensor(arg1_idxs).to(device) if device else torch.LongTensor(arg1_idxs)
    return raw


def dnee_ee_features(rels, model, config, pred2idx, argw2idx, max_event_len, rel2idx, device=None):
    x1_idx = 0
    x2_idx = 0
    gold2e1xs = {}
    gold2e2xs = {}
    x1s, x2s = [], []
    arg_lens = [config['arg0_max_len'], config['arg1_max_len']]
    for i_rel, rel in enumerate(rels):
        s = rel['Sense'][0]
        if len(rel['Arg1']['Events']) == 0:
            continue
        
        e1s = unique_event_dict(rel['Arg1']['Events'], pred2idx).values()
        for e1 in e1s:
            e1r = get_raw_event_repr(e1, config, pred2idx, argw2idx, device)
            x1s.append(e1r)
            if i_rel in gold2e1xs:
                gold2e1xs[i_rel].append(x1_idx)
            else:
                gold2e1xs[i_rel] = [x1_idx]
            x1_idx += 1

        e2s = unique_event_dict(rel['Arg2']['Events'], pred2idx).values()
        for e2 in e2s:
            e2r = get_raw_event_repr(e2, config, pred2idx, argw2idx, device)
            x2s.append(e2r)
            if i_rel in gold2e2xs:
                gold2e2xs[i_rel].append(x2_idx)
            else:
                gold2e2xs[i_rel] = [x2_idx]
            x2_idx += 1

    x1s = torch.stack(x1s, dim=0).squeeze()
    x2s = torch.stack(x2s, dim=0).squeeze()
    if device:
        x1s = x1s.to(device)
        x2s = x2s.to(device)
    with torch.no_grad():
        x1ee = model.embed_event(x1s)
        x2ee = model.embed_event(x2s)

    x1_out = torch.zeros((len(rels), max_event_len, x1ee.shape[1]), dtype=torch.float32)
    x2_out = torch.zeros((len(rels), max_event_len, x2ee.shape[1]), dtype=torch.float32)
    y = torch.LongTensor(len(rels))
    if device:
        x1_out = x1_out.to(device)
        x2_out = x2_out.to(device)
        y = y.to(device)
    for i_rel, rel in enumerate(rels):
        s = rel['Sense'][0]
        y[i_rel] = rel2idx[s]
        
        # combine scores for multiple event pairs
        if i_rel in gold2e1xs:
            idxs = gold2e1xs[i_rel]
            fs = x1ee[idxs, :]
            if fs.shape[0] > max_event_len:
                fs = fs[:max_event_len, :]
            x1_out[i_rel, :fs.shape[0]] = fs
        
        if i_rel in gold2e2xs:
            idxs = gold2e2xs[i_rel]
            fs = x2ee[idxs, :]
            if fs.shape[0] > max_event_len:
                fs = fs[:max_event_len, :]
            x2_out[i_rel, :fs.shape[0]] = fs
    return x1_out, x2_out, y


def unique_event_dict(edict, pred2idx):
    out = OrderedDict()
    for mid, es in six.iteritems(edict):
        for e in es:
            tmp_e = Event.from_json(e)
            if not tmp_e.valid_pred(pred2idx):
                continue
            if repr(tmp_e) not in out:
                out[repr(tmp_e)] = tmp_e
    return out


def dnee_score_features(rels, model, config, pred2idx, argw2idx, rel2idx, device=None):
    x_idx = 0
    gold2xs = {}
    xs = []
    arg_lens = [config['arg0_max_len'], config['arg1_max_len']]
    for i_rel, rel in enumerate(rels):
        s = rel['Sense'][0]
        if len(rel['Arg1']['Events']) == 0:
            continue
        
        e1s = unique_event_dict(rel['Arg1']['Events'], pred2idx).values()
        e2s = unique_event_dict(rel['Arg2']['Events'], pred2idx).values()
        for e1 in e1s:
            for e2 in e2s:
                erel = EventRelation(e1, e2, s)
                # if not erel.is_valid(pred2idx, config):
                    # continue
                if not erel.valid_pred(pred2idx):
                    continue
                raw = erel.to_indices(pred2idx, argw2idx, use_head=False, arg_len=arg_lens[0])
                x = EventRelationDataset.raw2x(raw, arg_lens)
                xs.append(x)
                if i_rel in gold2xs:
                    gold2xs[i_rel].append(x_idx)
                else:
                    gold2xs[i_rel] = [x_idx]
                x_idx += 1
    xs = torch.stack(xs, dim=0).squeeze()
    if device:
        xs = xs.to(device)
    scores = _scores_by_relations(model, xs, len(rel2idx), device)
    
    x_out = torch.zeros((len(rels), len(rel2idx)), dtype=torch.float32)
    y = torch.LongTensor(len(rels))
    if device:
        x_out = x_out.to(device)
        y = y.to(device)
    for i_rel, rel in enumerate(rels):
        s = rel['Sense'][0]
        y[i_rel] = rel2idx[s]
        
        # combine scores for multiple event pairs
        if i_rel in gold2xs:
            idxs = gold2xs[i_rel]
            fs = scores[:, idxs]
            if fs.shape[1] > 1:
                n = fs.shape[1]
                fs = torch.sum(fs, dim=1) / n
            x_out[i_rel] = fs.squeeze()
    return x_out, y


def rel_output(rel, predicted_sense):
    new_rel = {}
    new_rel['DocID'] = rel['DocID']
    new_rel['ID'] = rel['ID']

    new_rel['Arg1'] = {}
    new_rel['Arg1']['TokenList'] = []
    for tok in rel['Arg1']['TokenList']:
        new_rel['Arg1']['TokenList'].append(tok[2])

    new_rel['Arg2'] = {}
    new_rel['Arg2']['TokenList'] = []
    for tok in rel['Arg2']['TokenList']:
        new_rel['Arg2']['TokenList'].append(tok[2])
        
    new_rel['Connective'] = {}
    new_rel['Connective']['TokenList'] = []
    for tok in rel['Connective']['TokenList']:
        new_rel['Connective']['TokenList'].append(tok[2])
    
    new_rel['Sense'] = [predicted_sense]
    new_rel['Type'] = rel['Type']
    return new_rel


def create_cm(rels, rel2idx):
    valid_senses = set()
    sense_alphabet = Alphabet()
    for i, rel in enumerate(rels):
        s = rel['Sense'][0]
        if s in rel2idx:
            sense_alphabet.add(s)
            valid_senses.add(s)
    sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)
    cm = ConfusionMatrix(sense_alphabet)
    return cm, valid_senses


def scoring_cm(y, y_pred, cm, valid_senses, idx2rel):
    assert y.shape == y.shape
    cm.matrix.fill(0)
    for gold, pred in zip(y.tolist(), y_pred.tolist()):
        gold_s, pred_s = idx2rel[gold], idx2rel[pred]
        if gold_s not in valid_senses:
            gold_s = ConfusionMatrix.NEGATIVE_CLASS
        if pred_s not in valid_senses:
            pred_s = ConfusionMatrix.NEGATIVE_CLASS
        cm.add(pred_s, gold_s)
    prec, recall, f1 = cm.compute_micro_average_f1()
    return prec, recall, f1


def process_dev(fpath, arg_lens, pred2idx, argw2idx, config, rel2idx, device):
    rels = [json.loads(line) for line in open(fpath)]
    # we only care about implicit
    rels = [rel for rel in rels if rel['Type'] != 'Explicit' and rel['Sense'][0] in rel2idx]

    valid_senses = set()
    sense_alphabet = Alphabet()
    xs = []
    backoff_idxs = []
    gold2xs = {}
    x_idx = 0
    for i, rel in enumerate(rels):
        s = rel['Sense'][0]
        if s in rel2idx:
            sense_alphabet.add(s)
            valid_senses.add(s)

        if len(rel['Arg1']['Events']) == 0:
            backoff_idxs.append(i)
            continue

        # build features
        rtype = rel['Sense'][0] # for prediction it doesn't matter
        e1s = []
        for mid, es in six.iteritems(rel['Arg1']['Events']):
            for e in es:
                e1s.append(e)
        e2s = []
        for mid, es in six.iteritems(rel['Arg2']['Events']):
            for e in es:
                e2s.append(e)
        sub_xs = []
        for e1 in e1s:
            for e2 in e2s:
                ev1 = Event.from_json(e1)
                ev2 = Event.from_json(e2)
                erel = EventRelation(ev1, ev2, rtype)
                if not erel.is_valid(pred2idx, config):
                    continue

                raw = erel.to_indices(pred2idx, argw2idx, use_head=False)
                x = EventRelationDataset.raw2x(raw, arg_lens)
                sub_xs.append(x)
                if i in gold2xs:
                    gold2xs[i].append(x_idx)
                else:
                    gold2xs[i] = [x_idx]
                x_idx += 1
        if len(sub_xs) == 0:
            backoff_idxs.append(i)
        else:
            xs = xs + sub_xs
    xs = torch.stack(xs, dim=0).squeeze().to(device)

    sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)
    cm = ConfusionMatrix(sense_alphabet)
    return xs, gold2xs, backoff_idxs, rels, cm, valid_senses


def _scores_by_relations(model, xs, n_rels, device=None):
    scores = torch.zeros((n_rels, xs.shape[0]), dtype=torch.float32)
    if device:
        scores = scores.to(device)
    with torch.no_grad():
        for ridx in range(n_rels):
            xs[:, 0] = torch.LongTensor([ridx] * xs.shape[0])
            score = model(xs)
            scores[ridx] = score
    return scores


def _predict(model, xs, n_rels):
    with torch.no_grad():
        scores = torch.zeros((n_rels, xs.shape[0]), dtype=torch.float32)
        for ridx in range(n_rels):
            xs[:, 0] = torch.LongTensor([ridx] * xs.shape[0])
            score = model(xs)
            scores[ridx] = score.cpu()
        _, y_predict = torch.min(scores, 0)
    return y_predict


def _model_backoff(rel, rel2idx):
    return rel2idx['Expansion.Conjunction']


def scoring(gold2xs, backoff_idxs, rels, y_pred, n_rels, cm, valid_senses, rel2idx, idx2rel):
    cm.matrix.fill(0)
    for i, rel in enumerate(rels):
        if i in gold2xs:
            idxs = gold2xs[i]
            preds = y_pred[idxs]

            max_cnt = -1
            final_pred = -1
            for j in range(n_rels):
                cnt = (preds == j).sum().item()
                if cnt > max_cnt:
                    max_cnt = cnt
                    final_pred = j
        else:
            final_pred = _model_backoff(rel, rel2idx)

        pred_answer = idx2rel[final_pred]
        if pred_answer not in valid_senses:
            pred_answer = ConfusionMatrix.NEGATIVE_CLASS
        cm.add(pred_answer, rel['Sense'][0])
    prec, recall, f1 = cm.compute_micro_average_f1()
    return f1


def _eval(model, data, n_rels, rel2idx, idx2rel):
    xs, gold2xs, backoff_idxs, rels, cm, valid_senses = data
    y_pred = _predict(model, xs, n_rels)
    micro_f1 = scoring(gold2xs, backoff_idxs, rels, y_pred, n_rels, cm, valid_senses, rel2idx, idx2rel)
    return micro_f1


class DiscourseScorer(torch.nn.Module):
    def __init__(self, config, arg_dim, dnee_model, dnee_rel2idx, ds_rel2idx, use_we=True):
        super(DiscourseScorer, self).__init__()
        self.rel_embeddings = torch.nn.Embedding(len(ds_rel2idx), config['rel_dim'])
        self.init_rel_embeddings(dnee_model, dnee_rel2idx, config)
        
        self.use_we = use_we
        self.a1_linear = torch.nn.Linear(arg_dim, config['rel_dim'])
        self.a2_linear = torch.nn.Linear(arg_dim, config['rel_dim'])
        self.h1_linear = torch.nn.Linear(config['rel_dim']*3, config['rel_dim'])
        self.h2_linear = torch.nn.Linear(config['rel_dim'], 1)

    def init_rel_embeddings(self, dnee_model, dnee_rel2idx, ds_rel2idx, config):
        tmp_weights = torch.FloatTensor(len(ds_rel2idx), config['rel_dim'])
        torch.nn.init.xavier_uniform(tmp_weights)
        if len(dnee_rel2idx) == 6:
            dnee_sense_map = DNEE_SENSE_MAP_4REL
        elif len(dnee_rel2idx) == 11:
            dnee_sense_map = DNEE_SENSE_MAP_9REL
        else:
            raise ValueError('unsupported DNEE SENSE mapping')

        for task_rel, dnee_rel in six.iteritems(dnee_sense_map):
            task_idx = ds_rel2idx[task_rel]
            dnee_idx = torch.LongTensor([dnee_rel2idx[dnee_rel]])
            tmp_weights[task_idx] = dnee_model.rel_embeddings(dnee_idx)

        self.rel_embeddings.weight = torch.nn.Parameter(tmp_weights)

    def forward(self, dnee_model, x_e1, x_e2, x_r, x_w1=None, x_w2=None):
        e1 = dnee_model._transfer(x_e1, x_r)
        e2 = dnee_model._transfer(x_e2, x_r)
        x_a1 = e1 if x_w1 is None else torch.cat((e1, x_w1), 1)
        x_a2 = e2 if x_w2 is None else torch.cat((e2, x_w2), 1)

        a1 = F.relu(self.a1_linear(x_a1))
        a2 = F.relu(self.a2_linear(x_a2))
        rel = self.rel_embeddings(x_r)
        x = torch.cat((a1, a2, rel), 1)
        
        h1 = F.relu(self.h1_linear(x))
        out = self.h2_linear(h1)
        return out


class DiscourseSenseDataset(Dataset):
    def __init__(self, train_pos_data, train_neg_data):
        self.pos_e1, self.pos_e2, self.pos_r, self.pos_w1, self.pos_w2, self.pos_y = train_pos_data
        self.neg_e1, self.neg_e2, self.neg_r, self.neg_w1, self.neg_w2, self.neg_y = train_neg_data

    def __len__(self):
        return self.pos_e1.shape[0]

    def __getitem__(self, idx):
        return (self.pos_e1[idx], self.pos_e2[idx], self.pos_r[idx], self.pos_w1[idx], self.pos_w2[idx], self.pos_y[idx]), \
                (self.neg_e1[idx], self.neg_e2[idx], self.neg_r[idx], self.neg_w1[idx], self.neg_w2[idx], self.neg_y[idx])


def load_dataset(f, valid_senses):
    pdtb_file = open(f, 'r')
    relations = []
    cnt_invalid = 0
    for line in pdtb_file:
        rel = json.loads(line)
        sense = rel['Sense'][0]
        if sense not in valid_senses:
            cnt_invalid += 1
            continue
        relations.append(rel)
    pdtb_file.close()
    logging.info('cnt_invalid={}'.format(cnt_invalid))
    return relations


def dump_dataset(f, rels):
    out_file = open(f, 'w')
    for rel in rels:
        jstr = json.dumps(rel) + '\n' 
        out_file.write(jstr)
    out_file.close()


