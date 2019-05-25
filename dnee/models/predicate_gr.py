import os
import logging
from collections import OrderedDict
from ast import literal_eval

import six
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import progressbar
from scipy import spatial
import parse

from ..events import indices
from ..events.predicate_gr import Event
from .. import utils


def index_predicategr_from_event(e, e2idx):
    idxs = []
    key = '{}:{}'.format(e.pred, e.dep)
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e.arg0_head)
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e.arg1_head)
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e.arg2_head)
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    return idxs


event_parser = None
def index_predicategr_from_estr(estr, dep, e2idx):
    global event_parser
    if event_parser is None:
        event_parser = parse.compile("{pred}({arg0},{arg1},{arg2})")

    e = event_parser.parse(estr)
    key = '{}:{}'.format(e['pred'], dep)
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e['arg0'])
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e['arg1'])
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    key = 'arg:{}'.format(e['arg2'])
    idx = e2idx[key] if key in e2idx else 0
    idxs.append(idx)
    return idxs


def index_predicategr_example(e1, e2, nestr, e2idx):
    global event_parser
    idxs = []
    idxs += index_predicategr_from_event(e1, e2idx)
    idxs += index_predicategr_from_event(e2, e2idx)
    idxs += index_predicategr_from_estr(nestr, e2.dep, e2idx)
    return idxs


def load_word_embeddings(fpath, use_torch=False, skip_first_line=True):
    we = {}
    with open(fpath, 'r') as fr:
        for line in fr:
            if skip_first_line:
                skip_first_line = False
                continue
            line = line.rstrip()
            sp = line.split(" ")
            emb = np.squeeze(np.array([sp[1:]], dtype=np.float32))
            we[sp[0]] = torch.from_numpy(emb) if use_torch else emb
    return we


class PredicateGrBase(object):
    def __init__(self, **kwargs):
        pass

    def score(self, v1, v2):
        raise NotImplementedError
        
    def cosine_similarity(self, v1, v2):
        if v1.sum() == 0 and v2.sum() == 0:
            return 1.0
        return 1.0 - spatial.distance.cosine(v1, v2)

    def embed_event(self, e):
        raise NotImplementedError

    def predict_mcnc(self, ctx_events, choices):
        raise NotImplementedError


class EventComp(nn.Module, PredicateGrBase):
    def __init__(self, config, verbose=True, dropout=0.3):
        super(EventComp, self).__init__()
        self.event_dim = config['event_dim']
        self.embedding_file = config['embedding_file']
        self.e2idx = self.idx2e = None
        self.load_word2vec_event_emb()

        # dimensions are form the original paper
        self.arg_comp1_a1 = nn.Linear(self.event_dim*4, 600)
        self.tanh1_a1 = nn.Tanh()
        self.d1_a1 = nn.Dropout(p=dropout)
        self.arg_comp1_a2 = nn.Linear(self.event_dim*4, 600)
        self.tanh1_a2 = nn.Tanh()
        self.d1_a2 = nn.Dropout(p=dropout)

        self.arg_comp2_a1 = nn.Linear(600, 300)
        self.tanh2_a1 = nn.Tanh()
        self.d2_a1 = nn.Dropout(p=dropout)
        self.arg_comp2_a2 = nn.Linear(600, 300)
        self.tanh2_a2 = nn.Tanh()
        self.d2_a2 = nn.Dropout(p=dropout)

        self.event_comp1 = nn.Linear(600, 400)
        self.tanh_e1 = nn.Tanh()
        self.d_e1 = nn.Dropout(p=dropout)
        self.event_comp2 = nn.Linear(400, 200)
        self.tanh_e2 = nn.Tanh()
        self.d_e2 = nn.Dropout(p=dropout)
        self.event_comp3 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.arg_comp1_a1.weight.data)
        nn.init.xavier_uniform_(self.arg_comp1_a2.weight.data)
        nn.init.xavier_uniform_(self.arg_comp2_a1.weight.data)
        nn.init.xavier_uniform_(self.arg_comp2_a2.weight.data)
        nn.init.xavier_uniform_(self.event_comp1.weight.data)
        nn.init.xavier_uniform_(self.event_comp2.weight.data)
        nn.init.xavier_uniform_(self.event_comp3.weight.data)
    
    def predict_mcnc(self, ctx_events, choices):
        # the e2idx is not used
        all_idxs = []
        for i_ch, ch in enumerate(choices):
            ch_idx = index_predicategr_from_event(ch, self.e2idx)
            epairs = []
            for i_ctx, ctx in enumerate(ctx_events):
                ctx_idx = index_predicategr_from_event(ctx, self.e2idx)
                epairs.append(ch_idx + ctx_idx)
            all_idxs.append(epairs)
        all_idxs = torch.LongTensor(all_idxs)
        n_pair_per_ch = all_idxs.shape[1]
        input_dim = 8
        assert all_idxs.shape[2] == input_dim
        all_idxs = all_idxs.view(-1, input_dim)

        scores = self.forward(all_idxs).view(len(choices), n_pair_per_ch)
        scores = torch.sum(scores, dim=1) / n_pair_per_ch
        _, pred = torch.max(scores, 0)
        return pred.item()

    def save(self, fpath):
        torch.save(self.state_dict(), fpath)

    def load(self, fld):
        fpath = os.path.join(fld, 'model.pt')
        self.load_state_dict(torch.load(fpath, map_location=lambda storage, location: storage))

    def load_word2vec_event_emb(self):
        self.e2idx, self.idx2e = {}, {}
        n_embs = len([line for line in open(self.embedding_file)])
        embs = np.zeros((n_embs+1, self.event_dim))
        # leave index 0 as zero vector
        with open(self.embedding_file, 'r') as fr:
            for i, line in enumerate(fr):
                line = line.rstrip('\n')
                sp = line.split(" ")
                if i == 0:
                    self.n_embs = int(sp[0])
                    assert self.event_dim == int(sp[1])
                    continue
                ns = filter(None, sp[1:])
                ns = [float(n) for n in ns]
                emb = np.array(ns, dtype=np.float32)
                embs[i] = emb
                self.e2idx[sp[0]] = i
                self.idx2e[i] = sp[0]
        self.embs = nn.Embedding.from_pretrained(torch.FloatTensor(embs))

    def forward(self, x):
        batch_size = x.shape[0]
        x_embs = self.embs(x)
        e1 = x_embs[:, :4].view(batch_size, -1)
        e2 = x_embs[:, 4:8].view(batch_size, -1)

        out1_e1 = self.d1_a1(self.tanh1_a1(self.arg_comp1_a1(e1)))
        out1_e1 = self.d2_a1(self.tanh2_a1(self.arg_comp2_a1(out1_e1)))
        out1_e2 = self.d1_a2(self.tanh1_a2(self.arg_comp1_a2(e2)))
        out1_e2 = self.d2_a2(self.tanh2_a2(self.arg_comp2_a2(out1_e2)))

        epair = torch.cat((out1_e1, out1_e2), dim=1)
        
        out2 = self.d_e1(self.tanh_e1(self.event_comp1(epair)))
        out2 = self.d_e2(self.tanh_e2(self.event_comp2(out2)))
        out2 = self.sigmoid(self.event_comp3(out2))
        return out2.squeeze()
    
    def loss_func(self, cohs, eps=1e-12, lambda_l2=1e-3):
        pos_coh, neg_coh = cohs
        m = pos_coh.shape[0]
        loss = torch.sum(-torch.log(pos_coh + eps) - torch.log(1 - neg_coh + eps)) / m
        # l2 doesn't work well here, let's do dropout
        # l2_reg = None
        # for w in self.parameters():
            # if l2_reg is None:
                # l2_reg = torch.sum(w ** 2)
            # else:
                # l2_reg += torch.sum(w ** 2)
        # loss += (lambda_l2 * l2_reg)
        return loss


class Word2Vec(PredicateGrBase):
    def __init__(self, config, verbose=True):
        super(Word2Vec, self).__init__()
        fpath = config['embedding_file']
        self.embs = load_word_embeddings(fpath, skip_first_line=config['emb_file_skip_first_line'])
        self.dim = self.embs[self.embs.keys()[0]].shape[0]
    
    def score(self, v1, v2):
        return self.cosine_similarity(v1, v2)

    def aggr_emb(self, events):
        emb = np.zeros(self.dim, dtype=np.float32)
        cnt = 0
        for e in events:
            if e.pred in self.embs:
                emb += self.embs[e.pred]
                cnt += 1
            if e.arg0_head in self.embs:
                emb += self.embs[e.arg0_head]
                cnt += 1
            if e.arg1_head in self.embs:
                emb += self.embs[e.arg1_head]
                cnt += 1
            if e.arg2_head in self.embs:
                emb += self.embs[e.arg2_head]
                cnt += 1
        return emb if cnt > 0 else np.random.uniform(
                low=-1.0/self.dim, high=1.0/self.dim, size=self.dim)

    def predict_mcnc(self, ctx_events, choices):
        ctx_emb = self.aggr_emb(ctx_events)

        ch_embs = []
        for ch in choices:
            ch_emb = self.aggr_emb([ch])
            ch_embs.append(ch_emb)

        max_score = -1
        pred = -1
        scores = [0.0] * len(choices)
        for i, ch in enumerate(choices):
            scores[i] = self.score(ctx_emb, ch_embs[i])
            if scores[i] > max_score:
                max_score = scores[i]
                pred = i
        return pred


class Word2VecEvent(PredicateGrBase):
    def __init__(self, config, verbose=True):
        super(Word2VecEvent, self).__init__()
        fpath = config['embedding_file']
        self.embs = load_word_embeddings(fpath)
        self.dim = self.embs[self.embs.keys()[0]].shape[0]

    def _make_predicate_key(self, pred, dep):
        return "{}:{}".format(pred, dep)
    
    def _make_arg_key(self, arg):
        return "arg:{}".format(arg)

    def get_event_emb(self, e):
        """summation
        """
        emb = np.zeros(self.dim, dtype=np.float32)
        es = []
        key = self._make_predicate_key(e.pred, e.dep)
        if key in self.embs:
            emb += self.embs[key]
            es.append(key)
        key = self._make_arg_key(e.arg0_head)
        if key in self.embs:
            emb += self.embs[key]
            es.append(key)
        key = self._make_arg_key(e.arg1_head)
        if key in self.embs:
            emb += self.embs[key]
            es.append(key)
        key = self._make_arg_key(e.arg2_head)
        if key in self.embs:
            emb += self.embs[key]
            es.append(key)
        return es, emb

    def score(self, v1, v2):
        return self.cosine_similarity(v1, v2)

    def predict_mcnc(self, ctx_events, choices):
        ctx_es = []
        for ctx in ctx_events:
            es, ctx_emb = self.get_event_emb(ctx)
            ctx_es += es
        logging.debug(ctx_es)

        ch_embs = []
        for i, ch in enumerate(choices):
            ch_es, ch_emb = self.get_event_emb(ch)
            logging.debug("ch {}: {}".format(i, ch_es))
            ch_embs.append(ch_emb)

        max_score = -1
        pred = -1
        scores = [0.0] * len(choices)
        for i, ch in enumerate(choices):
            scores[i] = self.score(ctx_emb, ch_embs[i])
            if scores[i] > max_score:
                max_score = scores[i]
                pred = i
        return pred


class BiGram(PredicateGrBase):
    def __init__(self, params, verbose=True):
        super(BiGram, self).__init__()
        # do it later
        raise NotImplementedError


class CJ08(PredicateGrBase):
    def __init__(self, config, verbose=True):
        super(CJ08, self).__init__()
        self.verbose = verbose

        self.adj_m_fld = config['adj_matrix_folder']
        # self.adj_m = self.load_adj_m(os.path.join(adj_fld, 'adj_m.txt'))
        self.e2idx, self.idx2e = utils.load_indices(os.path.join(self.adj_m_fld, 'index_file.txt'))
        self.ppmi_m = None

    @staticmethod
    def load_adj_m(fpath, use_float=False):
        m = {}
        with open(fpath) as fr:
            for line in fr:
                line = line.rstrip('\n')
                sp = line.split('\t')
                if use_float:
                    m[literal_eval(sp[0])] = float(sp[1])
                else:
                    m[literal_eval(sp[0])] = int(sp[1])
        return m

    def train_ppmi(self, efreqs, adj_m ,e2idx, idx2e):
        self.ppmi_m = self._ppmi_matrix(efreqs, adj_m, e2idx, idx2e)

    def _adj_total_freq(self, m):

        dsum = m.diagonal().sum()
        upper = (m.sum() - dsum) / 2.0
        return upper + dsum

    def _ppmi_matrix(self, efreqs, adj_m ,e2idx, idx2e):
        ppmi_m = {}
        n_combs = sum(adj_m.values())
        freq_sum = sum([f for e, f in six.iteritems(efreqs)])

        if self.verbose:
            logging.info("learning PPMI...")
            widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=len(adj_m)).start()
            cnt = 0
        for k, v in six.iteritems(adj_m):
            idx1, idx2 = k
            if (idx2, idx1) in ppmi_m:
                continue
            v2 = adj_m[(idx2, idx1)] if (idx2, idx1) in adj_m else 0
            p_joint = float(v + v2) / n_combs
            e1, e2 = idx2e[idx1], idx2e[idx2]
            f1, f2 = efreqs[e1], efreqs[e2]
            p1 = float(f1) / freq_sum
            p2 = float(f2) / freq_sum
            ppmi_m[k] = ppmi_m[(idx2, idx1)] = np.log(p_joint / (p1 * p2))
            if self.verbose:
                cnt += 1
                bar.update(cnt)
        if self.verbose:
            bar.finish()
        return ppmi_m

    def _score_epair(self, estr0, estr1):
        assert estr0 in self.e2idx
        assert estr1 in self.e2idx
        idx0, idx1 = e2idx[estr0], e2idx[estr1]
        return self.ppmi_m[idx0, idx1]

    def save(self, fld):
        fpath = os.path.join(fld, 'ppmi.txt')
        with open(fpath, 'w') as fw:
            for k, v in six.iteritems(self.ppmi_m):
                fw.write('{}\t{}\n'.format(k, v))

    def load(self, fld):
        fpath = os.path.join(fld, 'ppmi.txt')
        logging.info('loading {}...'.format(fpath))
        self.ppmi_m = self.load_adj_m(fpath, use_float=True)
        fpath = os.path.join(self.adj_m_fld, 'index_file.txt')
        logging.info('loading {}...'.format(fpath))
        self.e2idx, self.idx2e = utils.load_indices(os.path.join(self.adj_m_fld, 'index_file.txt'))

    def predict_mcnc(self, ctx_events, choices):
        ctx_idxs = []
        for ctx in choices:
            estr = Event.cj08_format(ctx.pred, ctx.dep)
            if estr in self.e2idx:
                ctx_idxs.append(self.e2idx[estr])

        ch_estrs = []
        for ch in choices:
            estr = Event.cj08_format(ch.pred, ch.dep)
            ch_estrs.append(estr)

        max_score = float('-inf')
        pred = 0
        scores = [0.0] * len(ch_estrs)
        for i, ch in enumerate(ch_estrs):
            if ch in self.e2idx:
                cidx = self.e2idx[ch]
                scores[i] = sum([self.ppmi_m[(cidx, idx)] for idx in ctx_idxs if (cidx, idx) in self.ppmi_m])
            if scores[i] > max_score:
                max_score = scores[i]
                pred = i
        return pred


class SGNN(nn.Module):
    def __init__(self, config):
        super(SGNN, self).__init__()
        n_preds = self.get_n_preds(config['predicate_indices'])
        self.vocab = self.load_argw_vocabs(config['argw_indices'])
        self.save = True
        self.dir_st = config['skipthought_dir']
        self.embeddings = self._load_embeddings()
        import pdb; pdb.set_trace()
    
    @staticmethod
    def get_n_preds(fpath):
        pred2idx, idx2pred, _  = indices.load_predicates(fpath)
        return len(pred2idx)

    def _load_dictionary(self):
        path_dico = os.path.join(self.dir_st, 'dictionary.txt')
        with open(path_dico, 'r') as handle:
            dico_list = handle.readlines()
        dico = {word.strip():idx for idx,word in enumerate(dico_list)}
        return dico

    def _get_table_name(self):
        return 'btable'

    def _load_emb_params(self):
        table_name = self._get_table_name()
        path_params = os.path.join(self.dir_st, table_name+'.npy')
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
    
    def _make_emb_state_dict(self, dictionary, parameters):
        weight = torch.zeros(len(self.vocab)+1, 620) # first dim = zeros -> +1
        unknown_params = parameters[dictionary['UNK']]
        nb_unknown = 0
        for id_weight, word in enumerate(self.vocab):
            if word in dictionary:
                id_params = dictionary[word]
                params = parameters[id_params]
            else:
                #print('Warning: word `{}` not in dictionary'.format(word))
                params = unknown_params
                nb_unknown += 1
            weight[id_weight+1] = torch.from_numpy(params)
        state_dict = OrderedDict({'weight':weight})
        if nb_unknown > 0:
            print('Warning: {}/{} words are not in dictionary, thus set UNK'
                  .format(nb_unknown, len(dictionary)))
        return state_dict
    
    def _load_embeddings(self, emb_fpath=None):
        if self.save:
            import hashlib
            import pickle
            # http://stackoverflow.com/questions/20416468/fastest-way-to-get-a-hash-from-a-list-in-python
            hash_id = hashlib.sha256(pickle.dumps(self.vocab, -1)).hexdigest()
            path = emb_fpath if emb_fpath else 'st_embedding_'+str(hash_id)+'.pth'
        if self.save and os.path.exists(path):
            self.embedding = torch.load(path)
        else:
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1,
                                          embedding_dim=620,
                                          padding_idx=0, # -> first_dim = zeros
                                          sparse=False)
            dictionary = self._load_dictionary()
            parameters = self._load_emb_params()
            state_dict = self._make_emb_state_dict(dictionary, parameters)
            self.embedding.load_state_dict(state_dict)
            if self.save:
                torch.save(self.embedding, path)
        return self.embedding
    
    def load_argw_vocabs(self, fpath):
        key2idx, _, _ = indices.load_argw(fpath)
        return key2idx.keys()
