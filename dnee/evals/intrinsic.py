import random
import re
import time
import logging
from copy import deepcopy

import six
import numpy as np
import torch
from sklearn.metrics import pairwise
from nltk import word_tokenize

from ..events import Event, indices


def embed_event_word_embeddings(e, we):
    # words = [w.lower() for w in word_tokenize(e.sentence)]
    tokens = [e.pred]
    arg0s = [w.lower() for w in word_tokenize(e.arg0)]
    arg1s = [w.lower() for w in word_tokenize(e.arg1)]
    arg2s = [w.lower() for w in word_tokenize(e.arg2)]
    tokens += arg0s + arg1s + arg2s
    dim = next(six.itervalues(we)).shape[0]
    avg = torch.zeros(dim, dtype=torch.float32)
    cnt = 0
    for t in tokens:
        if t not in we:
            continue
        avg += we[t]
        cnt += 1
    if cnt > 0:
        avg /= cnt
    return avg


class CorefQuestion:
    def __init__(self, query_event, ans_idx, choices):
        self.query_event = query_event
        self.choices = choices
        self.ans_idx = ans_idx
    
    @classmethod
    def from_doc(cls, echain, event_list, n_choices=5):
        excludes = {}
        for e in echain:
            excludes[e.__repr__()] = e

        # pick first event and one coreferenced event
        ridx = random.randint(1, len(echain)-1)
        query_event = echain[0]
        ans_event = echain[ridx]

        # pick 5 un-coreferenced events
        choices = []
        while len(choices) < n_choices:
            c = event_list[random.randint(0, len(event_list)-1)]
            if c.__repr__() not in excludes:
                choices.append(c)
                excludes[c.__repr__()] = c
        ans_idx = random.randint(0, n_choices-1)
        choices[ans_idx] = ans_event
        return cls(query_event, ans_idx, choices)


class CorefBinaryQuestion:
    def __init__(self, e1, e2, label):
        self.e1 = e1
        self.e2 = e2
        self.label = label

    def __repr__(self):
        return "{}, {}, {}".format(self.e1, self.e2, self.label)


class DiscourseQuestion:
    def __init__(self, rel, ans_idx, choices):
        self.rel = rel
        self.choices = choices
        self.ans_idx = ans_idx
    
    @classmethod
    def from_relation(cls, drel, epool, n_choices=5):
        ans_idx = random.randint(0, n_choices-1)
        choices = []
        while len(choices) < n_choices:
            c = epool[random.randint(0, len(epool)-1)]
            if c not in choices and c != drel.e1 and c != drel.e2:
                choices.append(c)
        choices[ans_idx] = drel.e2
        return cls(drel, ans_idx, choices)


class MCNCEvent(Event):
    def __init__(self, pred, dep, arg0, arg0_head, arg1, arg1_head, arg2, arg2_head,
                    sentiment, ani0, ani1, ani2, sent):
        super(MCNCEvent, self).__init__(pred, arg0, arg0_head, arg1, arg1_head,
                arg2, arg2_head, sentiment, ani0, ani1, ani2)
        self.dep = dep
        rep = {'\n': ' ', '::': ' '}
        rep = dict((re.escape(k), v) for k, v in rep.iteritems())
        pat = re.compile("|".join(rep.keys()))
        self.sentence = pat.sub(lambda m: rep[re.escape(m.group(0))], sent)
        if sys.version_info < (3, 0):
            self.sentence = self.sentence.encode('ascii', 'ignore')

    def __repr__(self):
        return "({}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{})".format(
                self.pred, self.dep,
                self.arg0, self.arg0_head,
                self.arg1, self.arg1_head,
                self.arg2, self.arg2_head,
                self.sentiment, self.ani0,
                self.ani1, self.ani2, self.sentence)

    @classmethod
    def from_string(cls, line):
        raise NotImplementedError
        # line = line.rstrip("\n")[1:-1]
        # sp = line.split('::')
        # obj = cls(sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6],
                    # sp[7], sp[8], sp[9], sp[10], sp[11], sp[12])
        # return obj

    @classmethod
    def from_json(cls, e):
        pred = e['predicate']
        # only use the first sub-argument for now
        arg0_head = e['arg0'][0] if 'arg0' in e else indices.NO_ARG
        arg0 = e['arg0_text'][0] if 'arg0_text' in e else indices.NO_ARG

        arg1_head = e['arg1'][0] if 'arg1' in e else indices.NO_ARG
        arg1 = e['arg1_text'][0] if 'arg1_text' in e else indices.NO_ARG

        arg2_head = e['arg2'][0] if 'arg2' in e else indices.NO_ARG
        arg2 = e['arg2_text'][0] if 'arg2_text' in e else indices.NO_ARG
        sentiment = e['sentiment'] if 'sentiment' in e else None
        ani0 = e['ani0'][0] if 'ani0' in e else indices.UNKNOWN_ANIMACY
        ani1 = e['ani1'][0] if 'ani1' in e else indices.UNKNOWN_ANIMACY
        ani2 = e['ani2'][0] if 'ani2' in e else indices.UNKNOWN_ANIMACY
        dep = e['dep']
        sent = e['sentence']
        obj = cls(pred, dep, arg0, arg0_head, arg1, arg1_head, arg2, arg2_head,
                    sentiment, ani0, ani1, ani2, sent)
        return obj


# MCNC Question
class Question:
    def __init__(self, q_idx, ans_idx, choices, echain):
        self.echain = echain
        self.q_idx = q_idx
        self.ans_idx = ans_idx
        self.choices = choices

    def get_contexts(self):
        contexts = self.echain[:self.q_idx]
        if len(self.echain) > self.q_idx+1:
            contexts += self.echain[self.q_idx+1:]
        return contexts
    
    def __repr__(self):
        return '{}, {}, {}, {}'.format(
                self.q_idx,
                self.ans_idx,
                self.choices,
                self.echain)

    @classmethod
    def from_event_chain(cls, echain, epool, n_choices=5, fixed_len=9):
        echain = echain[:fixed_len]
        ehash = {}
        for e in echain:
            ehash[e.__repr__()] = 1

        all_args = []
        # randomly pick shared arguments from echain
        for e in echain:
            if e.arg0_head != indices.NO_ARG:
                all_args.append((e.arg0_head, e.arg0))
            if e.arg1_head != indices.NO_ARG:
                all_args.append((e.arg1_head, e.arg1))
            if e.arg2_head != indices.NO_ARG:
                all_args.append((e.arg2_head, e.arg2))

        q_idx = fixed_len-1
        ans_idx = random.randint(0, n_choices-1)
        choices = []
        while len(choices) < n_choices:
            c = epool[random.randint(0, len(epool)-1)]
            new_c = deepcopy(c)

            # replace with protagonist
            rpos = random.randint(0, 2)
            rarg = all_args[random.randint(0, len(all_args)-1)]
            if rpos == 0:
                new_c.arg0_head, new_c.arg0 = rarg
            elif rpos == 1:
                new_c.arg1_head, new_c.arg1 = rarg
            else:
                new_c.arg2_head, new_c.arg2 = rarg

            if new_c.__repr__() not in ehash and new_c not in choices:
                choices.append(new_c)
        choices[ans_idx] = echain[q_idx]
        return cls(q_idx, ans_idx, choices, echain)


class MCNSQuestion:
    def __init__(self, echain, choice_lists, ans_idxs):
        self.echain = echain
        self.choice_lists = choice_lists
        self.ans_idxs = ans_idxs
        assert len(ans_idxs) == len(echain)-1
        assert len(ans_idxs) == len(choice_lists)

    def __repr__(self):
        return '{}, {}, {}'.format(
                self.ans_idxs,
                self.choice_lists,
                self.echain)

    @classmethod
    def from_event_chain(cls, echain, epool, n_choices=5, chain_len=5):
        ehash = {}
        for e in echain:
            ehash[e.__repr__()] = 1

        assert len(echain) >= chain_len
        subechain = echain[:chain_len]
        
        all_args = []
        # randomly pick shared arguments from echain
        for e in subechain:
            if e.arg0_head != indices.NO_ARG:
                all_args.append((e.arg0_head, e.arg0))
            if e.arg1_head != indices.NO_ARG:
                all_args.append((e.arg1_head, e.arg1))
            if e.arg2_head != indices.NO_ARG:
                all_args.append((e.arg2_head, e.arg2))

        choice_lists = []
        ans_idxs = []
        for i in range(1, chain_len):
            cs = []
            choice_hash = {}
            while len(cs) < n_choices:
                c = epool[random.randint(0, len(epool)-1)]
                new_c = deepcopy(c)

                # replace with protagonist
                rpos = random.randint(0, 2)
                rarg = all_args[random.randint(0, len(all_args)-1)]
                if rpos == 0:
                    new_c.arg0_head, new_c.arg0 = rarg
                elif rpos == 1:
                    new_c.arg1_head, new_c.arg1 = rarg
                else:
                    new_c.arg2_head, new_c.arg2 = rarg
                if new_c.__repr__() not in ehash and new_c.__repr__() not in choice_hash:
                    cs.append(new_c)
                    choice_hash[new_c.__repr__()] = 1
            ans_idx = random.randint(0, n_choices-1)
            cs[ans_idx] = subechain[i]
            choice_lists.append(cs)
            ans_idxs.append(ans_idx)
        return cls(subechain, choice_lists, ans_idxs)


def get_event_emb(model, e, e2idx, embeddings, device):
    idx = e2idx[e.__repr__()]
    emb = embeddings[idx]
    return emb


def predict_mcnc(model, q, e2idx, embeddings, rtype, device):
    contexts = q.get_contexts()
    ctx_embs = [get_event_emb(model, e, e2idx, embeddings, device)
                    for e in contexts]
    choice_embs = [get_event_emb(model, e, e2idx, embeddings, device)
                    for e in q.choices]
    rel_emb = model.rel_embeddings(rtype)
    # take avg of scores between event pairs
    energies = []
    for ch in choice_embs:
        sub_es = []
        for ctx in ctx_embs:
            e = model._calc(ch, ctx, rel_emb)
            e = torch.sum(e, 1) # L1-norm
            if model.norm > 1:
                e = torch.pow(e, 1.0 / model.norm)
            sub_es.append(e[0])
        avg = sum(sub_es) / len(sub_es)
        energies.append(avg)
    energies = torch.Tensor(energies).to(device)
    v, idx = energies.min(0)
    return idx


def scoring_cosine_similarity(emb1s, emb2s):
    cs = pairwise.cosine_similarity(emb1s, emb2s)
    # shift from [-1, 1] to [0, 1]
    ret = (cs + 1) / 2.0
    return ret


def _we_transition_prob(ts1, ts2):
    ts1_embs = torch.stack(ts1, dim=0)
    ts2_embs = torch.stack(ts2, dim=0)

    scores = pairwise.cosine_similarity(ts1_embs.cpu(), ts2_embs.cpu())
    # shift from [-1, 1] to [0, 1] to avoid negative values
    scores = (scores + 1) / 2.0
    
    probs = scores / scores.sum(axis=1, keepdims=True)
    assert (probs < 0).sum() == 0
    return torch.from_numpy(probs)


def _ev_transition_prob(ts1, ts2, rel_emb, model):
    tm = torch.zeros((len(ts1), len(ts2)), dtype=torch.float32)
    for i in range(len(ts1)):
        for j in range(len(ts2)):
            e = model._calc(ts1[i], ts2[j], rel_emb)
            e = torch.sum(e, 1) # L1-norm
            tm[i][j] = torch.pow(e, 1.0 / model.norm) if model.norm > 1 else e
    tm = 1.0 / tm
    probs = tm / tm.sum(1)
    assert (probs < 0).sum().tolist() == 0
    return probs


def transition_prob_matrix(model, q, e2idx, ev_embeddings, w_embeddings, rtype):
    timestamp_we_embs = [[w_embeddings[e2idx[q.echain[0].__repr__()]]]]
    timestamp_ev_embs = [[ev_embeddings[e2idx[q.echain[0].__repr__()]]]]
    for clist in q.choice_lists:
        choice_we_embs = [w_embeddings[e2idx[c.__repr__()]] for c in clist]
        choice_ev_embs = [ev_embeddings[e2idx[c.__repr__()]] for c in clist]
        timestamp_we_embs.append(choice_we_embs)
        timestamp_ev_embs.append(choice_ev_embs)

    rel_emb = model.rel_embeddings(rtype)
    # first transition
    all_we_probs, all_ev_probs = [], []
    we_probs = _we_transition_prob(timestamp_we_embs[0], timestamp_we_embs[1])
    ev_probs = _ev_transition_prob(timestamp_ev_embs[0], timestamp_ev_embs[1], rel_emb, model)
    all_we_probs.append(we_probs)
    all_ev_probs.append(ev_probs)

    # the rest
    for i in range(1, len(timestamp_ev_embs)-1):
        we_probs = _we_transition_prob(timestamp_we_embs[i], timestamp_we_embs[i+1])
        ev_probs = _ev_transition_prob(timestamp_ev_embs[i], timestamp_ev_embs[i+1], rel_emb, model)
        all_we_probs.append(we_probs)
        all_ev_probs.append(ev_probs)
    return all_we_probs, all_ev_probs


def transition_prob_matrix_all_orders(model, q, e2idx, ev_embeddings, w_embeddings, rtype):
    timestamp_we_embs = [[w_embeddings[e2idx[q.echain[0].__repr__()]]]]
    timestamp_ev_embs = [[ev_embeddings[e2idx[q.echain[0].__repr__()]]]]
    for clist in q.choice_lists:
        choice_we_embs = [w_embeddings[e2idx[c.__repr__()]] for c in clist]
        choice_ev_embs = [ev_embeddings[e2idx[c.__repr__()]] for c in clist]
        timestamp_we_embs.append(choice_we_embs)
        timestamp_ev_embs.append(choice_ev_embs)

    rel_emb = model.rel_embeddings(rtype)
    # first transition
    all_we_probs, all_ev_probs = [], []
    for i in range(len(timestamp_we_embs)):
        tmp_we, tmp_ev = [], []
        for j in range(len(timestamp_we_embs)):
            if i >= j:
                tmp_we.append(None)
                tmp_ev.append(None)
            else:
                _we = _we_transition_prob(timestamp_we_embs[i], timestamp_we_embs[j])
                _ev = _ev_transition_prob(timestamp_ev_embs[i], timestamp_ev_embs[j], rel_emb, model)
                tmp_we.append(_we)
                tmp_ev.append(_ev)
        all_we_probs.append(tmp_we)
        all_ev_probs.append(tmp_ev)
    return all_we_probs, all_ev_probs


def viterbi(tpms):
    n_choices = tpms[1].shape[0]
    trellis = torch.zeros((n_choices, len(tpms)), dtype=torch.float32)
    backtrace = torch.ones((n_choices, len(tpms)), dtype=torch.int64) * -1
    
    t1 = time.time()
    trellis[:, 0] = tpms[0]

    for t in range(1, len(tpms)):
        for j in range(n_choices):
            tmp_probs = torch.zeros(n_choices, dtype=torch.float32)
            for k in range(n_choices):
                p = trellis[k, t-1] * tpms[t][k][j]
                tmp_probs[k] = p
            trellis[j, t] = torch.max(tmp_probs)
            backtrace[j, t] = torch.argmax(tmp_probs)

   # backtrace
    tokens = [trellis[:, -1].argmax()]
    for i in xrange(len(tpms)-1, 0, -1):
        tokens.append(backtrace[tokens[-1], i])
    preds = tokens[::-1]
    logging.debug('viterbi: {} s'.format(time.time()-t1))
    return preds


def markov_baseline(tpms):
    # base on the previous prediction
    preds = []
    for t in range(len(tpms)):
        if t == 0:
            pred = torch.argmax(tpms[t])
        else:
            previous_state = preds[t-1]
            pred = torch.argmax(tpms[t][previous_state])
        preds.append(pred)
    return preds


def markov_skyline(tpms, ans_idxs, fix_end, n_choices=5):
    # base on the correct previous state
    # preds = []
    # for t in range(len(tpms)):
        # if t == 0:
            # pred = torch.argmax(tpms[t])
        # else:
            # previous_state = ans_idxs[t-1]
            # pred = torch.argmax(tpms[t][previous_state])
        # preds.append(pred)
    
    # consider all previous and future gold states
    # when predicting each step
    preds = []
    for target in range(1, len(tpms)):
        scores = torch.zeros(n_choices, dtype=torch.float32)
        for src in range(len(tpms)):
            if src == target:
                continue
            elif src < target:
                m = tpms[src][target]
                if src == 0:
                    scores += m.squeeze()
                else:
                    ans_idx = ans_idxs[src-1]
                    scores += m[ans_idx, :]
            else:
                m = tpms[target][src]
                ans_idx = ans_idxs[src-1]
                scores += m[:, ans_idx]
        _, pred = torch.max(scores, 0)
        preds.append(pred)
    return preds


def _predict_sequence(model, q, e2idx, ev_embeddings, w_embeddings, rtype, device, fix_end=False, inference_model='Viterbi'):
    if inference_model != 'Skyline':
        we_tpms, ev_tpms = transition_prob_matrix(model, q, e2idx, ev_embeddings, w_embeddings, rtype)
        # if fix end, we simply make the last tranition probs to zero, except the correct one
        if fix_end:
            last_we_tpm = we_tpms[-1]
            last_ev_tpm = ev_tpms[-1]
            for i in range(last_ev_tpm.shape[0]):
                for j in range(last_ev_tpm.shape[1]):
                    if j != q.ans_idxs[-1]:
                        last_we_tpm[i, j] = 0.0
                        last_ev_tpm[i, j] = 0.0
        # mix_tpms = [(we_tpms[i] + ev_tpms[i]) / 2.0 for i in range(len(we_tpms))]
    else:
        we_tpms, ev_tpms = transition_prob_matrix_all_orders(model, q, e2idx, ev_embeddings, w_embeddings, rtype)

    if inference_model == 'Viterbi':
        we_preds = viterbi(we_tpms)
        ev_preds = viterbi(ev_tpms)
        # mix_preds = viterbi(mix_tpms)
    elif inference_model == 'Baseline':
        we_preds = markov_baseline(we_tpms)
        ev_preds = markov_baseline(ev_tpms)
        # mix_preds = markov_baseline(mix_tpms)
    elif inference_model == 'Skyline':
        we_preds = markov_skyline(we_tpms, q.ans_idxs, fix_end)
        ev_preds = markov_skyline(ev_tpms, q.ans_idxs, fix_end)
        # mix_preds = markov_skyline(mix_tpms, q.ans_idxs, fix_end)
    else:
        raise ValueError('unsupported inference model {}'.format(args.inference_model))

    return we_preds, ev_preds


def predict_mcns(model, q, e2idx, ev_embeddings, w_embeddings, rtype, device, inference_model):
    return _predict_sequence(model, q, e2idx, ev_embeddings, w_embeddings, rtype, device, fix_end=False, inference_model=inference_model)


def predict_mcne(model, q, e2idx, ev_embeddings, w_embeddings, rtype, device, inference_model):
    return _predict_sequence(model, q, e2idx, ev_embeddings, w_embeddings, rtype, device, fix_end=True, inference_model=inference_model)
