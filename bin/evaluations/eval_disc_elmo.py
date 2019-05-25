import sys
import os
import logging
import argparse
import json
import time
import re
import pickle as pkl

import six
import numpy as np
import torch
import torch.nn.functional as F
import progressbar
from sklearn.metrics import accuracy_score, classification_report
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='intrinsic disc evaluation')
    parser.add_argument('question_file', metavar='QUESTION_FILE',
                        help='questions.')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='relation config')

    parser.add_argument('-w', '--elmo_weight_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
                        help='ELMo weight file')
    parser.add_argument('-p', '--elmo_option_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_options.json",
                        help='ELMo option file')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='batch size for evaluation')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def _eval(questions, we, eval_func):
    n_batches = len(questions) // args.batch_size
    if len(questions) % args.batch_size > 0:
        n_batches += 1

    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=n_batches).start()
    ys, y_preds = [], []
    for i_batch in range(n_batches):
        subquestions = questions[i_batch*args.batch_size: (i_batch+1)*args.batch_size]
        logging.debug("#subquestions = {}".format(len(subquestions)))
        y, y_pred = eval_func(subquestions, we)
        ys += y
        y_preds += y_pred
        bar.update(i_batch+1)
    bar.finish()
    return ys, y_preds


def _eval_by_rel_we(questions, we):
    ys, y_preds = [], []
    for q in questions:
        x1 = intrinsic.embed_event_word_embeddings(q.rel.e1, we)
        y = q.ans_idx
        ys.append(y)

        scores = torch.zeros(len(q.choices), dtype=torch.float32)
        for i, c in enumerate(q.choices):
            ch = intrinsic.embed_event_word_embeddings(c, we)
            scores[i] = utils.cosine_similarity(x1, ch)
        _, y_pred = torch.max(scores, 0)
        y_preds.append(y_pred.item())
    return ys, y_preds


def event_toks(e):
    preds = e.pred.split('_')
    arg0s = [w.lower() for w in word_tokenize(e.arg0)]
    arg1s = [w.lower() for w in word_tokenize(e.arg1)]
    arg2s = [w.lower() for w in word_tokenize(e.arg2)]
    return preds + arg0s + arg1s + arg2s


lemmatizer = None
def embed_event_elmo(etoks, tok2idx, embs, dim=512):
    global lemmatizer
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    lemma2idx = {lemmatizer.lemmatize(k, 'v'): v for k, v in six.iteritems(tok2idx)}
    cnt = 0
    res = torch.zeros(dim, dtype=torch.float32).to(args.device)
    for t in etoks:
        if t in tok2idx:
            res += embs[tok2idx[t]]
            cnt += 1
        elif t in lemma2idx:
            res += embs[lemma2idx[t]]
            cnt += 1
    if cnt > 0:
        res /= cnt
    return res


def get_event_sentence(e):
    # return e.sentence
    preds = ' '.join(e.pred.split('_'))
    sent = e.arg0 + ' ' + preds + ' ' + e.arg1
    return sent


mrr = 0.0
def _eval_by_rel_elmo(questions, we):
    n_choices = len(questions[0].choices)
    e1_sents, ch_sents = [], []
    for q in questions:
        e1_tokens = [w.lower() for w in word_tokenize(get_event_sentence(q.rel.e1))]
        e1_sents.append(e1_tokens)

        for i_c, ch in enumerate(q.choices):
            ch_tokens = [w.lower() for w in word_tokenize(get_event_sentence(ch))]
            ch_sents.append(ch_tokens)


    e1_ids = batch_to_ids(e1_sents).to(args.device)
    ch_ids = batch_to_ids(ch_sents).to(args.device)
    e1_embs = we(e1_ids)['elmo_representations'][0]
    ch_embs = we(ch_ids)['elmo_representations'][0]
    ys, y_preds = [], []
    for i_q, q in enumerate(questions):
        e1_sent = e1_sents[i_q]
        e1_tok2idx = {w: i for i, w in enumerate(e1_sent)}
        e1_toks = event_toks(q.rel.e1)
        e1_emb = embed_event_elmo(e1_toks, e1_tok2idx, e1_embs[i_q])

        scores = torch.zeros(n_choices, dtype=torch.float32).to(args.device)
        _ch_sents = ch_sents[i_q * n_choices: (i_q+1) * n_choices]
        _ch_embs = ch_embs[i_q * n_choices: (i_q+1) * n_choices]
        for i_c, ch in enumerate(q.choices):
            ch_sent = _ch_sents[i_c]
            tmp_ch_embs = _ch_embs[i_c]
            ch_tok2idx = {w: i for i, w in enumerate(ch_sent)}
            ch_toks = event_toks(ch)
            ch_emb = embed_event_elmo(ch_toks, ch_tok2idx, tmp_ch_embs)
            scores[i_c] = F.cosine_similarity(e1_emb, ch_emb, dim=0)
        
        _, y_pred = torch.max(scores, 0)
        y_preds.append(y_pred.item())
        ys.append(q.ans_idx)
    return ys, y_preds


def _eval_by_rel(questions, we):
    return _eval_by_rel_elmo(questions, we)


def eval_by_rel(questions, we):
    return _eval(questions, we, _eval_by_rel)


def main():
    indices.set_relation_classes(args.relation_config)
    questions = pkl.load(open(args.question_file, 'rb'))
    logging.info("#questions={}".format(len(questions)))

    we = Elmo(args.elmo_option_file, args.elmo_weight_file, 1, dropout=0).to(args.device)
    with torch.no_grad():
        logging.info('evaluating by rel')
        y, y_pred = eval_by_rel(questions, we)
        print("accuracy = {}".format(accuracy_score(y, y_pred)))
        logging.info("accuracy = {}".format(accuracy_score(y, y_pred)))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if args.gpu_id is not None:
        args.device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
