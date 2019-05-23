import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from .skipthoughts import UniSkip, BiSkip, CustomizedBiSkip
from ..events import indices

logger = logging.getLogger(__name__)

# These models are portings from OpenKE
# https://github.com/thunlp/OpenKE
# Thanks to their effort


class ArgWordEncoder(object):
    def __init__(self, config, argw_vocabs=None, use_uniskip=True, use_biskip=True, use_customized_biskip=False, device=torch.device('cpu')):
        self.device = device
        self.argw_vocabs = self.load_argw_vocabs(config['argw_indices']) if argw_vocabs is None else argw_vocabs
        self.output_dim = 0
        if use_uniskip:
            # fixed, no parameters
            self.uniskip = UniSkip(config['skipthought_dir'], self.argw_vocabs,
                                    save=True, dropout=0, fixed_emb=True,
                                    fixed_net=True, device=self.device).to(self.device)
            self.output_dim += 2400
        else:
            self.uniskip = None
        
        if use_biskip:
            # fixed, no parameters
            self.biskip = BiSkip(config['skipthought_dir'], self.argw_vocabs,
                                    save=True, dropout=0, fixed_emb=True,
                                    fixed_net=True, device=self.device).to(self.device) 
            self.output_dim += 2400
        else:
            self.biskip = None

        # network not fixed
        if use_customized_biskip:
            hidden_size = config['pred_dim'] / 2
            logger.info("cskip runs on {}, hidden size = {}".format(self.device, hidden_size))
            self.c_biskip = CustomizedBiSkip(config['skipthought_dir'], self.argw_vocabs,
                                    save=True, dropout=0.0, fixed_emb=True,
                                    fixed_net=False, device=self.device,
                                    hidden_size=hidden_size).to(self.device)
            self.output_dim += config['pred_dim']
        else:
            self.c_biskip = None
        logger.info("arg_dim: {}".format(self.output_dim))

    def encode(self, x):
        uvec = self.uniskip(x) if self.uniskip else None
        bvec = self.biskip(x) if self.biskip else None
        cvec = self.c_biskip(x) if self.c_biskip else None
        comb_vec = tuple(v for v in (uvec, bvec, cvec) if v is not None)
        comb_vec = torch.cat(comb_vec, 1)
        return comb_vec
    
    def save(self, fpath):
        if self.c_biskip:
            torch.save(self.c_biskip.state_dict(), fpath)
        # not saving skipthough, since won't update it

    def load(self, fpath):
        if self.c_biskip:
            self.c_biskip.load_state_dict(torch.load(fpath, map_location=lambda storage, loc: storage))

    @staticmethod
    def load_argw_vocabs(fpath):
        key2idx, _, _ = indices.load_argw(fpath)
        return list(key2idx.keys())


class AbstractEventTrans(nn.Module):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(AbstractEventTrans, self).__init__()
        self.norm = config['norm']
        self.device = device
        self.config = config
        self.n_preds = self.get_n_preds(config['predicate_indices']) if n_preds is None else n_preds
        self.argw_encoder = argw_encoder
        logging.info('pred_dim: {}'.format(config['pred_dim']))
        self.rel_zero_shot = rel_zero_shot
        if rel_zero_shot:
            logging.info('n_old_rel_types: {}'.format(config['n_old_rel_types']))
            self.rel_embeddings = nn.Embedding(config['n_old_rel_types'], config['rel_dim']).to(self.device)
        else:
            logging.info('n_rel_types: {}'.format(config['n_rel_types']))
            self.rel_embeddings = nn.Embedding(config['n_rel_types'], config['rel_dim']).to(self.device)

        logging.info('rel_dim: {}'.format(config['rel_dim']))
        self.pred_embeddings = nn.Embedding(self.n_preds, config['pred_dim']).to(self.device)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.pred_embeddings.weight.data)
    
    def transfer_rel(self, sense_map, new_rel2idx, old_rel2idx):
        assert self.rel_zero_shot
        new_rel_embeddings = nn.Embedding(self.config['n_rel_types'], self.config['rel_dim']).to(self.device)
        nn.init.xavier_uniform_(new_rel_embeddings.weight.data)
        for rel, new_idx in new_rel2idx.iteritems():
            if rel in sense_map:
                old_idx = old_rel2idx[sense_map[rel]]
                new_rel_embeddings.weight.data[new_idx] = self.rel_embeddings.weight.data[old_idx]
        self.rel_embeddings = new_rel_embeddings

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config['margin'], False).to(self.device)
        y = Variable(torch.Tensor([-1]).to(self.device), requires_grad=False)
        loss = criterion(p_score, n_score, y)
        return loss
    
    @staticmethod
    def get_n_preds(fpath):
        pred2idx, idx2pred, _  = indices.load_predicates(fpath)
        return len(pred2idx)

    def forward(self, x):
        raise NotImplementedError
   
    def _get_event_raw_features(self, p, arg0, arg1):
        p_emb = self.pred_embeddings(p)
        arg0_emb = self.argw_encoder.encode(arg0)
        arg1_emb = self.argw_encoder.encode(arg1)
        ev_raw = torch.cat((p_emb, arg0_emb, arg1_emb), 1)
        return ev_raw

    def _event_indices(self, x):
        assert x.shape[1] == 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        p = x[:, 0]
        arg0 = x[:, 1:1+self.config['arg0_max_len']]
        arg1 = x[:, 1+self.config['arg0_max_len']: 1+self.config['arg0_max_len']+self.config['arg1_max_len']]
        return p, arg0, arg1

    def _calc(self, h, t, r):
        return torch.abs(h + r - t) if self.norm == 1 else torch.pow(torch.abs(h + r - t), self.norm)


class AbstractCompEventTrans(AbstractEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(AbstractCompEventTrans, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)
        event_raw_dim = config['pred_dim'] + self.argw_encoder.output_dim * 2
        logging.info('event_raw_dim: {}'.format(event_raw_dim))
        logging.info('event_hidden_dim: {}'.format(config['event_hidden_dim']))
        self.event_l1 = nn.Linear(event_raw_dim, config['event_hidden_dim'])
        self.relu = nn.ReLU()
        self.event_l2 = nn.Linear(config['event_hidden_dim'], config['event_dim'])
        logging.info('event_dim: {}'.format(config['event_dim']))
    
    def embed_event(self, x):
        ev_len = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        assert ev_len == x.shape[1]
        ev_p, ev_arg0, ev_arg1 = self._event_indices(x)
        ev_raw = self._get_event_raw_features(ev_p, ev_arg0, ev_arg1)
        ev_h1 = self.relu(self.event_l1(ev_raw))
        ev_emb = self.event_l2(ev_h1)
        return ev_emb
    
    def forward(self, x):
        raise NotImplementedError


class AbstractAttnEventTrans(AbstractEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(AbstractAttnEventTrans, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)
        assert config['pred_dim'] == self.argw_encoder.output_dim
        assert config['pred_dim'] == config['event_dim']
        self.w_attn = nn.Parameter(torch.FloatTensor(config['pred_dim'], 1))
        nn.init.xavier_normal_(self.w_attn)

    def embed_event(self, x):
        ev_len = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        assert ev_len == x.shape[1]
        ev_p, ev_arg0, ev_arg1 = self._event_indices(x)
        ev_raw = self._get_event_raw_features(ev_p, ev_arg0, ev_arg1)
        batch_size = x.shape[0]
        d = self.config['pred_dim']
        # attention: a soft combination of v_pred, v_a0, v_a1
        v_pred, v_arg0, v_arg1 = ev_raw[:, :d], \
                                    ev_raw[:, d:2*d], \
                                    ev_raw[:, 2*d:]
        vs = torch.stack((v_pred, v_arg0, v_arg1), dim=1)
        attn_scores = torch.matmul(vs, self.w_attn).squeeze()
        attn_scores = F.softmax(attn_scores).view(batch_size, 3, 1)
        scored_vs = vs * attn_scores
        condensed_vs = torch.sum(scored_vs, dim=1)
        return condensed_vs

    def forward(self, x):
        raise NotImplementedError
    

class EventTransE(AbstractCompEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(EventTransE, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)

    def forward(self, x):
        rtype = x[:, 0]
        rel_emb = self.rel_embeddings(rtype)

        elen = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        e1_emb = self.embed_event(x[:, 1: 1+elen])
        e2_emb = self.embed_event(x[:, 1+elen: 1+elen+elen])

        _score = self._calc(e1_emb, e2_emb, rel_emb)
        score = torch.sum(_score, 1)
        if self.norm > 1:
            score = torch.pow(score, 1.0 / self.norm)
        return score

    def _transfer(self, ev_emb, rtype):
        return ev_emb


class EventTransR(AbstractCompEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(EventTransR, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)
        self.transfer_matrix = nn.Embedding(config['n_rel_types'], config['rel_dim']*config['event_dim']).to(self.device)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
    
    def forward(self, x):
        rtype = x[:, 0]
        rel_emb = self.rel_embeddings(rtype).view(-1, self.config['rel_dim'])

        rel_m = self.transfer_matrix(rtype).view(-1, self.config['rel_dim'], self.config['event_dim'])
        
        elen = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        e1_emb = self.embed_event(x[:, 1: 1+elen]).view(-1, self.config['event_dim'], 1)
        e2_emb = self.embed_event(x[:, 1+elen: 1+elen+elen]).view(-1, self.config['event_dim'], 1)

        e1_emb_t = torch.matmul(rel_m, e1_emb).view(-1, self.config['rel_dim'])
        e2_emb_t = torch.matmul(rel_m, e2_emb).view(-1, self.config['rel_dim'])
        
        _score = self._calc(e1_emb_t, e2_emb_t, rel_emb)
        score = torch.sum(_score, 1)
        if self.norm > 1:
            score = torch.pow(score, 1.0 / self.norm)
        return score
    
    def _transfer(self, ev_emb, rtype):
        if rtype.shape == torch.Size([1]):
            rtype = rtype.expand(ev_emb.shape[0], 1)
        m = self.transfer_matrix(rtype).view(-1, self.config['rel_dim'], self.config['event_dim'])
        ev_emb = ev_emb.view(-1, self.config['event_dim'], 1)
        return torch.matmul(m, ev_emb).view(-1, self.config['rel_dim'])


class AttnEventTransE(AbstractAttnEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(AttnEventTransE, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)

    def forward(self, x):
        rtype = x[:, 0]
        rel_emb = self.rel_embeddings(rtype)

        elen = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        e1_emb = self.embed_event(x[:, 1: 1+elen])
        e2_emb = self.embed_event(x[:, 1+elen: 1+elen+elen])

        _score = self._calc(e1_emb, e2_emb, rel_emb)
        score = torch.sum(_score, 1)
        if self.norm > 1:
            score = torch.pow(score, 1.0 / self.norm)
        return score

    def _transfer(self, ev_emb, rtype):
        return ev_emb


class AttnEventTransR(AbstractAttnEventTrans):
    def __init__(self, config, argw_encoder, n_preds=None, device=torch.device('cpu'), rel_zero_shot=False):
        super(AttnEventTransR, self).__init__(config, argw_encoder, n_preds, device, rel_zero_shot)
        self.transfer_matrix = nn.Embedding(config['n_rel_types'], config['rel_dim']*config['event_dim']).to(self.device)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
    
    def forward(self, x):
        rtype = x[:, 0]
        rel_emb = self.rel_embeddings(rtype).view(-1, self.config['rel_dim'])

        rel_m = self.transfer_matrix(rtype).view(-1, self.config['rel_dim'], self.config['event_dim'])
        
        elen = 1 + self.config['arg0_max_len'] + self.config['arg1_max_len']
        e1_emb = self.embed_event(x[:, 1: 1+elen]).view(-1, self.config['event_dim'], 1)
        e2_emb = self.embed_event(x[:, 1+elen: 1+elen+elen]).view(-1, self.config['event_dim'], 1)

        e1_emb_t = torch.matmul(rel_m, e1_emb).view(-1, self.config['rel_dim'])
        e2_emb_t = torch.matmul(rel_m, e2_emb).view(-1, self.config['rel_dim'])
        
        _score = self._calc(e1_emb_t, e2_emb_t, rel_emb)
        score = torch.sum(_score, 1)
        if self.norm > 1:
            score = torch.pow(score, 1.0 / self.norm)
        return score
    
    def _transfer(self, ev_emb, rtype):
        if rtype.shape == torch.Size([1]):
            rtype = rtype.expand(ev_emb.shape[0], 1)
        m = self.transfer_matrix(rtype).view(-1, self.config['rel_dim'], self.config['event_dim'])
        ev_emb = ev_emb.view(-1, self.config['event_dim'], 1)
        return torch.matmul(m, ev_emb).view(-1, self.config['rel_dim'])


def create_argw_encoder(config, device):
    logging.info('argw_encoder: {}'.format(config['argw_encoder_opt']))
    argw_vocabs = ArgWordEncoder.load_argw_vocabs(config['argw_indices'])
    if config['argw_encoder_opt'] == 'customized_biskip':
        argw_encoder = ArgWordEncoder(config, argw_vocabs,
                use_uniskip=False, use_biskip=False, use_customized_biskip=True, device=device)
    elif config['argw_encoder_opt'] == 'skipthoughts':
        argw_encoder = ArgWordEncoder(config, argw_vocabs,
                use_uniskip=True, use_biskip=True, use_customized_biskip=False, device=device)
    elif config['argw_encoder_opt'] == 'uniskip':
        argw_encoder = ArgWordEncoder(config, argw_vocabs,
                use_uniskip=True, use_biskip=False, use_customized_biskip=False, device=device)
    elif config['argw_encoder_opt'] == 'biskip':
        argw_encoder = ArgWordEncoder(config, argw_vocabs,
                use_uniskip=False, use_biskip=True, use_customized_biskip=False, device=device)
    else:
        raise ValueError('unsupported encoder type.')
    return argw_encoder
