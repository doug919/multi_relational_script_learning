import sys
import os
import logging
import argparse
import json
import time
import re
import cPickle as pkl

import numpy as np
import torch
import progressbar
from sklearn.metrics import f1_score, accuracy_score

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder
from dnee.models import AttnEventTransE, AttnEventTransR


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='intrinsic disc evaluation')
    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='model file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='encoder file.')
    parser.add_argument('question_file', metavar='QUESTION_FILE',
                        help='questions.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='relation classes')

    parser.add_argument('-b', '--batch_size', type=int, default=500,
                        help='batch size for evaluation')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def build_examples(questions, config, pred2idx, argw2idx):
    Xs, ys = [], []
    for q in questions:
        x1 = utils.get_raw_event_repr(q.rel.e1, config, pred2idx, argw2idx)
        x2 = utils.get_raw_event_repr(q.rel.e2, config, pred2idx, argw2idx)
        x = torch.cat((x1, x2), 0)
        Xs.append(x)
        y = q.rel.rtype_idx
        ys.append(y)
    Xs = torch.stack(Xs, dim=0).to(args.device)
    ys = torch.LongTensor(ys).to(args.device)
    return Xs, ys


def _eval_by_events(questions, model, config, pred2idx, argw2idx, relation_config):
    global tmp_y, tmp_y_pred
    X, y = build_examples(questions, config, pred2idx, argw2idx)
    ev_dim = X.shape[1] // 2
    e1 = X[:, :ev_dim]
    e2 = X[:, ev_dim:]
    embs1 = model.embed_event(e1)
    embs2 = model.embed_event(e2)
    
    # predict
    n_rels = relation_config['disc_end'] - relation_config['disc_begin']
    scores = torch.zeros((n_rels, X.shape[0]), dtype=torch.float32)
    for r in range(n_rels):
        ridx = torch.LongTensor([r]*e1.shape[0]).to(args.device)
        remb = model.rel_embeddings(ridx)
        _score = model._calc(model._transfer(embs1, ridx),
                                model._transfer(embs2, ridx),
                                remb)
        score = torch.sum(_score, 1)
        if model.norm > 1:
            score = torch.pow(score, 1.0 / model.norm)
        scores[r] = score
    _, y_predict = torch.min(scores, 0)
    return y, y_predict


def eval_by_events(questions, model, config, pred2idx, argw2idx, relation_config):
    return _eval(questions, model, config,
                    pred2idx, argw2idx, _eval_by_events, relation_config)


def _eval(questions, model, config, pred2idx, argw2idx, eval_func, relation_config):
    n_batches = len(questions) // args.batch_size
    if len(questions) % args.batch_size > 0:
        n_batches += 1

    n_correct, n_incorrect = 0, 0
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=n_batches).start()
    ys, y_preds = [], []
    for i_batch in range(n_batches):
        subquestions = questions[i_batch*args.batch_size: (i_batch+1)*args.batch_size]
        logging.debug("#subquestions = {}".format(len(subquestions)))
        y, y_pred = eval_func(subquestions, model, config, pred2idx, argw2idx, relation_config)
        ys.append(y)
        y_preds.append(y_pred)
        bar.update(i_batch+1)
    ys = torch.cat(ys, dim=0).detach().cpu().tolist()
    y_preds = torch.cat(y_preds, dim=0).detach().cpu().tolist()
    bar.finish()
    return ys, y_preds


def _eval_by_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    embs1, ridxs, ys = [], [], []
    all_cins = []
    for q in questions:
        x1 = utils.get_raw_event_repr(q.rel.e1, config, pred2idx, argw2idx)
        embs1.append(x1)
        ridxs.append(q.rel.rtype_idx)
        y = q.ans_idx
        ys.append(y)

        cins = []
        for c in q.choices:
            cin = utils.get_raw_event_repr(c, config, pred2idx, argw2idx)
            cins.append(cin.tolist())
        all_cins.append(cins)

    embs1 = torch.stack(embs1, dim=0).to(args.device)
    ridxs = torch.LongTensor(ridxs).to(args.device)
    ys = torch.LongTensor(ys).to(args.device)
    embs1 = model._transfer(model.embed_event(embs1), ridxs)
    rembs = model.rel_embeddings(ridxs)

    all_cins = torch.LongTensor(all_cins).to(args.device)
    n_choices = len(questions[0].choices)
    scores = torch.zeros((n_choices, len(questions)), dtype=torch.float32)
    for i in range(n_choices):
        cembs = model.embed_event(all_cins[:, i, :])
        _score = model._calc(embs1,
                                model._transfer(cembs, ridxs),
                                rembs)
        score = torch.sum(_score, 1)
        if model.norm > 1:
            score = torch.pow(score, 1.0 / model.norm)
        scores[i] = score
    _, y_predict = torch.min(scores, 0)
    return ys, y_predict


def _eval_by_next_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    embs1, ridxs, ys = [], [], []
    all_cins = []
    for q in questions:
        x1 = utils.get_raw_event_repr(q.rel.e1, config, pred2idx, argw2idx)
        embs1.append(x1)
        ridxs.append(relation_config["rel2idx"][indices.REL_CONTEXT])
        y = q.ans_idx
        ys.append(y)

        cins = []
        for c in q.choices:
            cin = utils.get_raw_event_repr(c, config, pred2idx, argw2idx)
            cins.append(cin.tolist())
        all_cins.append(cins)
    
    embs1 = torch.stack(embs1, dim=0).to(args.device)
    ridxs = torch.LongTensor(ridxs).to(args.device)
    ys = torch.LongTensor(ys).to(args.device)
    embs1 = model._transfer(model.embed_event(embs1), ridxs)
    rembs = model.rel_embeddings(ridxs)

    all_cins = torch.LongTensor(all_cins).to(args.device)
    n_choices = len(questions[0].choices)
    scores = torch.zeros((n_choices, len(questions)), dtype=torch.float32)
    for i in range(n_choices):
        cembs = model.embed_event(all_cins[:, i, :])
        _score = model._calc(embs1,
                                model._transfer(cembs, ridxs),
                                rembs)
        score = torch.sum(_score, 1)
        if model.norm > 1:
            score = torch.pow(score, 1.0 / model.norm)
        scores[i] = score
        
    _, y_predict = torch.min(scores, 0)
    return ys, y_predict


def _eval_by_random_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    embs1, ridxs, ys = [], [], []
    all_cins = []
    for q in questions:
        x1 = utils.get_raw_event_repr(q.rel.e1, config, pred2idx, argw2idx)
        embs1.append(x1)
        ridxs.append(q.rel.rtype_idx)
        y = q.ans_idx
        ys.append(y)

        cins = []
        for c in q.choices:
            cin = utils.get_raw_event_repr(c, config, pred2idx, argw2idx)
            cins.append(cin.tolist())
        all_cins.append(cins)

    embs1 = torch.stack(embs1, dim=0).to(args.device)
    ridxs = torch.LongTensor(ridxs).to(args.device)
    ys = torch.LongTensor(ys).to(args.device)
    embs1 = model._transfer(model.embed_event(embs1), ridxs)
    rridxs = torch.randint(relation_config['disc_begin'], relation_config['disc_end'], ridxs.shape, dtype=torch.int64).to(args.device)
    rrembs = model.rel_embeddings(rridxs)

    all_cins = torch.LongTensor(all_cins).to(args.device)
    n_choices = len(questions[0].choices)
    rscores = torch.zeros((n_choices, len(questions)), dtype=torch.float32)
    for i in range(n_choices):
        cembs = model.embed_event(all_cins[:, i, :])
        _rscore = model._calc(embs1,
                                model._transfer(cembs, rridxs),
                                rrembs)
        rscore = torch.sum(_rscore, 1)
        if model.norm > 1:
            rscore = torch.pow(rscore, 1.0 / model.norm)
        rscores[i] = rscore

    _, ry_predict = torch.min(rscores, 0)
    return ys, ry_predict


def eval_by_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    return _eval(questions, model, config,
                    pred2idx, argw2idx, _eval_by_rel, relation_config)

    
def eval_by_random_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    return _eval(questions, model, config,
                    pred2idx, argw2idx, _eval_by_random_rel, relation_config)


def eval_by_next_rel(questions, model, config, pred2idx, argw2idx, relation_config):
    return _eval(questions, model, config,
                    pred2idx, argw2idx, _eval_by_next_rel, relation_config)


def acc(y, y_pred):
    a = np.array(y, dtype=np.int64)
    b = np.array(y_pred, dtype=np.int64)
    n_correct = (a == b).sum()
    return float(n_correct) / a.shape[0]


def main():
    config = json.load(open(args.training_config, 'r'))
    relation_config = json.load(open(args.relation_config, 'r'))
    pred2idx, idx2pred, _  = indices.load_predicates(config['predicate_indices'])
    argw2idx, idx2argw, _  = indices.load_argw(config['argw_indices'])
    n_preds = len(pred2idx)
    argw_vocabs = argw2idx.keys()
    argw_encoder = create_argw_encoder(config, args.device)
    if args.encoder_file:
        argw_encoder.load(args.encoder_file)
    
    logging.info("model class: " + config['model_type'])
    ModelClass = eval(config['model_type'])
    model = ModelClass(config, argw_encoder, n_preds, args.device).to(args.device)
    model.load_state_dict(torch.load(args.model_file,
                                    map_location=lambda storage, location: storage))
    
    questions = pkl.load(open(args.question_file, 'r'))
    logging.info("#questions={}".format(len(questions)))
    
    logging.info('predict relation')
    y, y_pred = eval_by_events(questions, model, config, pred2idx, argw2idx, relation_config)
    logging.info("predict relation, accuracy={}".format(accuracy_score(y, y_pred)))
    logging.info("predict relation, accuracy={}".format(acc(y, y_pred)))

    logging.info('predict next event')
    y, y_pred = eval_by_rel(questions, model, config, pred2idx, argw2idx, relation_config)
    logging.info("predict next event, accuracy={}".format(accuracy_score(y, y_pred)))
    logging.info("predict next event, accuracy={}".format(acc(y, y_pred)))

    logging.info('predict next event by random rel')
    y, y_pred = eval_by_random_rel(questions, model, config, pred2idx, argw2idx, relation_config)
    logging.info("by random rel, accuracy={}".format(accuracy_score(y, y_pred)))
    logging.info("by random rel, accuracy={}".format(acc(y, y_pred)))
    
    logging.info('predict next event by next rel')
    y, y_pred = eval_by_next_rel(questions, model, config, pred2idx, argw2idx, relation_config)
    logging.info("by next rel, accuracy={}".format(accuracy_score(y, y_pred)))
    logging.info("by next rel, accuracy={}".format(acc(y, y_pred)))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if args.gpu_id is not None:
        args.device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
