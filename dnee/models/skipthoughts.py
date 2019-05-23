"""
    a porting from https://github.com/Cadene/skip-thoughts.torch
    thanks to their effort.
"""

import os
import sys
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict


###############################################################
# UniSkip
###############################################################
class AbstractSkipThoughts(nn.Module):

    def __init__(self, dir_st, vocab, save=True, dropout=0, fixed_emb=True,
                    fixed_net=True, emb_fpath=None, device=torch.device('cpu')):
        super(AbstractSkipThoughts, self).__init__()
        self.device = device
        self.dir_st = dir_st
        self.vocab = vocab
        self.save = save
        self.dropout = dropout
        self.fixed_emb = fixed_emb
        # Module
        self.embedding = self._load_embedding(emb_fpath)
        if fixed_emb:
            self.embedding.weight.requires_grad = False
        self.rnn = self._load_rnn()
        if fixed_net:
            for p in self.rnn.parameters():
                p.requires_grad = False

    def _get_table_name(self):
        raise NotImplementedError

    def _get_skip_name(self):
        raise NotImplementedError

    def _load_dictionary(self):
        path_dico = os.path.join(self.dir_st, 'dictionary.txt')
        with open(path_dico, 'r') as handle:
            dico_list = handle.readlines()
        dico = {word.strip():idx for idx,word in enumerate(dico_list)}
        return dico

    def _load_emb_params(self):
        table_name = self._get_table_name()
        path_params = os.path.join(self.dir_st, table_name+'.npy')
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params
 
    def _load_rnn_params(self):
        skip_name = self._get_skip_name()
        path_params = os.path.join(self.dir_st, skip_name+'.npz')
        params = numpy.load(path_params, encoding='latin1') # to load from python2
        return params

    def _load_embedding(self, emb_fpath):
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

    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i]-1].fill_(1)
        mask = Variable(mask)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, -1)
        return x

    def _select_last_old(self, input, lengths):
        batch_size = input.size(0)
        x = []
        for i in range(batch_size):
            x.append(input[i,lengths[i]-1].view(1, -1))
        output = torch.cat(x, 0)
        return output
 
    def _process_lengths(self, input):
        max_length = input.size(1)
        if input.shape[0] == 1:
            lengths = (max_length - input.data.eq(0).sum(1)).tolist()
        else:
            lengths = (max_length - input.data.eq(0).sum(1).squeeze()).tolist()
        return lengths

    def _load_rnn(self):
        raise NotImplementedError

    def _make_rnn_state_dict(self, p):
        raise NotImplementedError

    def forward(self, input, lengths=None):
        raise NotImplementedError


###################################################################################
# UniSkip
###################################################################################

class UniSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=True, dropout=0.0, fixed_emb=True,
                    fixed_net=True, emb_fpath=None, device=torch.device('cpu')):
        super(UniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb, fixed_net, emb_fpath, device)
        # Remove bias_ih_l0 (== zero all the time)
        # del self.gru._parameters['bias_hh_l0']
        # del self.gru._all_weights[0][3]

    def _get_table_name(self):
        return 'utable'

    def _get_skip_name(self):
        return 'uni_skip'

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620,
                          hidden_size=2400,
                          batch_first=True,
                          dropout=self.dropout)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0']   = torch.zeros(7200) 
        s['bias_hh_l0']   = torch.zeros(7200) # must stay equal to 0
        s['weight_ih_l0'] = torch.zeros(7200, 620)
        s['weight_hh_l0'] = torch.zeros(7200, 2400)
        s['weight_ih_l0'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:4800]   = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][4800:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][4800:] = torch.from_numpy(p['encoder_Ux']).t()             
        return s

    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        x, hn = self.rnn(x) # seq2seq
        if lengths:
            x = self._select_last(x, lengths)
        return x



###############################################################
# BiSkip
###############################################################

class BiSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=True, dropout=0.0, fixed_emb=True,
                    fixed_net=True, emb_fpath=None, device=torch.device('cpu')):
        super(BiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb, fixed_net, emb_fpath, device)
        # Remove bias_ih_l0 (== zero all the time)
        # del self.gru._parameters['bias_hh_l0']
        # del self.gru._all_weights[0][3]

    def _get_table_name(self):
        return 'btable'

    def _get_skip_name(self):
        return 'bi_skip'

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620,
                          hidden_size=1200,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0']   = torch.zeros(3600) 
        s['bias_hh_l0']   = torch.zeros(3600) # must stay equal to 0
        s['weight_ih_l0'] = torch.zeros(3600, 620)
        s['weight_hh_l0'] = torch.zeros(3600, 1200)

        s['bias_ih_l0_reverse']   = torch.zeros(3600) 
        s['bias_hh_l0_reverse']   = torch.zeros(3600) # must stay equal to 0
        s['weight_ih_l0_reverse'] = torch.zeros(3600, 620)
        s['weight_hh_l0_reverse'] = torch.zeros(3600, 1200)
        
        s['weight_ih_l0'][:2400] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][2400:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:2400]   = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][2400:]   = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:2400] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][2400:] = torch.from_numpy(p['encoder_Ux']).t()  

        s['weight_ih_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_W']).t()
        s['weight_ih_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Wx']).t()
        s['bias_ih_l0_reverse'][:2400]   = torch.from_numpy(p['encoder_r_b'])
        s['bias_ih_l0_reverse'][2400:]   = torch.from_numpy(p['encoder_r_bx'])
        s['weight_hh_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_U']).t()
        s['weight_hh_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Ux']).t() 
        return s

    def _argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        if lengths is None:
            lengths = self._process_lengths(input)
        sorted_lengths = sorted(lengths)
        sorted_lengths = sorted_lengths[::-1]
        idx = self._argsort(lengths)
        idx = idx[::-1] # decreasing order
        inverse_idx = self._argsort(idx)
        idx = Variable(torch.LongTensor(idx))
        inverse_idx = Variable(torch.LongTensor(inverse_idx))
        if input.data.is_cuda:
            idx = idx.to(self.device)
            inverse_idx = inverse_idx.to(self.device)
        x = torch.index_select(input, 0, idx)

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
        x, hn = self.rnn(x) # seq2seq
        hn = hn.transpose(0, 1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, 2 * hn.size(2))

        hn = torch.index_select(hn, 0, inverse_idx)
        return hn


###############################################################
# RawBiSkip
###############################################################

class CustomizedBiSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=True, dropout=0.0, fixed_emb=True,
                    fixed_net=False, emb_fpath=None, device=torch.device('cpu'),
                    hidden_size=250):
        self.hidden_size = hidden_size
        super(CustomizedBiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb, fixed_net, emb_fpath, device)
        # Remove bias_ih_l0 (== zero all the time)
        # del self.gru._parameters['bias_hh_l0']
        # del self.gru._all_weights[0][3]

    def _get_table_name(self):
        return 'btable'

    def _get_skip_name(self):
        return 'bi_skip'

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620,
                          hidden_size=int(self.hidden_size),
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)
        return self.rnn

    def _argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        if lengths is None:
            lengths = self._process_lengths(input)
        sorted_lengths = sorted(lengths)
        sorted_lengths = sorted_lengths[::-1]
        idx = self._argsort(lengths)
        idx = idx[::-1] # decreasing order
        inverse_idx = self._argsort(idx)
        idx = Variable(torch.LongTensor(idx))
        inverse_idx = Variable(torch.LongTensor(inverse_idx))
        if input.data.is_cuda:
            idx = idx.to(self.device)
            inverse_idx = inverse_idx.to(self.device)
        x = torch.index_select(input, 0, idx)

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
        x, hn = self.rnn(x) # seq2seq
        hn = hn.transpose(0, 1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, 2 * hn.size(2))

        hn = torch.index_select(hn, 0, inverse_idx)
        return hn


if __name__ == '__main__':
    dir_st = 'data/skipthought_models'
    vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']

    us_model = UniSkip(dir_st, vocab)
    bs_model = BiSkip(dir_st, vocab)

    # batch_size x seq_len
    input = Variable(torch.LongTensor([
        [6,1,2,3,3,4,0],
        [6,1,2,3,3,4,5],
        [1,2,3,4,0,0,0]
    ]))
    print(input.size())

    # for skipping dropout layers
    us_model.eval()
    bs_model.eval()

    # batch_size x 2400
    us_seq2vec = us_model(input)
    print(us_seq2vec)

    bs_seq2vec = bs_model(input)
    print(bs_seq2vec)
