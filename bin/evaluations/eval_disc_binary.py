import sys
import os
import logging
import argparse
import json
import time
import re
import pickle as pkl
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F
import progressbar
from sklearn import metrics
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk import word_tokenize

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='relation specific binary classification')
    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='model file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='encoder file.')
    parser.add_argument('dev_question_file', metavar='DEV_QUESTION_FILE',
                        help='dev questions.')
    parser.add_argument('test_question_file', metavar='TEST_QUESTION_FILE',
                        help='test questions.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='relation classes')

    parser.add_argument('-w', '--elmo_weight_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
                        help='ELMo weight file')
    parser.add_argument('-p', '--elmo_option_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_options.json",
                        help='ELMo option file')
    parser.add_argument('-m', '--use_elmo', action='store_true', default=False,
                        help='use ELMo but TransE/TransR')
    parser.add_argument('-r', '--n_rounds', type=int, default=10,
                        help='number of rounds to get the average')
    parser.add_argument('-s', '--step_size', type=float, default=0.001,
                        help='grid search step size for thresholds')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def sample_questions(question_file, n_cat_questions=500):
    questions = pkl.load(open(question_file, 'rb'))

    all_epairs = {}
    all_events = {}
    cat_questions = {}
    for q in questions:
        if q.rel.e1.__repr__() not in all_events:
            all_events[q.rel.e1.__repr__()] = q.rel.e1
        if q.rel.e2.__repr__() not in all_events:
            all_events[q.rel.e2.__repr__()] = q.rel.e1

        
        if q.rel.rtype_idx not in all_epairs:
            all_epairs[q.rel.rtype_idx] = {}
        key = str((q.rel.e1, q.rel.e2))
        all_epairs[q.rel.rtype_idx][key] = 1
        key = str((q.rel.e2, q.rel.e1))
        all_epairs[q.rel.rtype_idx][key] = 1

        if q.rel.rtype_idx not in cat_questions:
            cat_questions[q.rel.rtype_idx] = []
        cat_questions[q.rel.rtype_idx].append((1, q))
    
    for i in range(len(cat_questions)):
        cat_questions[i] = cat_questions[i][:n_cat_questions]

    # negative
    evs = list(all_events.values())
    for i_cat in range(len(cat_questions)):
        nqs = []
        for label, q in cat_questions[i_cat]:
            nq = deepcopy(q)
            nq.choices = None
            nq.ans_idx = None
            while True:
                tmpe = evs[np.random.randint(0, len(evs))]
                key = str((nq.rel.e1, tmpe))
                if key not in all_epairs[q.rel.rtype_idx]:
                    nq.rel.e2 = tmpe
                    break
            nqs.append((0, nq))
        cat_questions[i_cat] += (nqs)
    return cat_questions


def build_ev_embeddings(questions, config, pred2idx, argw2idx, model):
    with torch.no_grad():
        idx =  0
        e2idx = {}
        xs = []
        for i_cat in questions.keys():
            for label, q in questions[i_cat]:
                for e in [q.rel.e1, q.rel.e2]:
                    if e.__repr__() not in e2idx:
                        e2idx[e.__repr__()] = idx
                        idx += 1
                        x = utils.get_raw_event_repr(e, config, pred2idx, argw2idx)
                        xs.append(x)
        xs = torch.stack(xs, dim=0).to(args.device)
        embeddings = model.embed_event(xs)
    return e2idx, embeddings


def get_event_sentence(e):
    # return e.sentence
    preds = ' '.join(e.pred.split('_'))
    sent = e.arg0 + ' ' + preds + ' ' + e.arg1
    return sent


def _elmo_batch(ids, elmo, batch_size=500):
    with torch.no_grad():
        embss = []
        n_batches = ids.shape[0] // batch_size
        if ids.shape[0] % batch_size > 0:
            n_batches += 1
        for i_batch in range(n_batches):
            start = i_batch * batch_size
            end = (i_batch + 1) * batch_size
            
            embs = elmo(ids[start:end])['elmo_representations'][0].detach()
            embss.append(embs)
        embss = torch.cat(embss, dim=0).sum(dim=1)
    return embss


def build_elmo(questions, elmo):
    idx =  0
    e2idx = {}
    xs = []
    for i_cat in questions.keys():
        for label, q in questions[i_cat]:
            for e in [q.rel.e1, q.rel.e2]:
                if e.__repr__() not in e2idx:
                    e2idx[e.__repr__()] = idx
                    idx += 1

                    s = get_event_sentence(e)
                    e_tokens = [w.lower() for w in word_tokenize(s)]
                    xs.append(e_tokens)
    xs = batch_to_ids(xs).to(args.device)
    embeddings = _elmo_batch(xs, elmo)
    return e2idx, embeddings


def score_questions(questions, model, e2idx, embeddings):
    with torch.no_grad():
        cat_scores = {}
        for i_cat in questions.keys():
            e1s, e2s, rs = [], [], []
            for label, q in questions[i_cat]:
                e1 = embeddings[e2idx[q.rel.e1.__repr__()]]
                e2 = embeddings[e2idx[q.rel.e2.__repr__()]]
                e1s.append(e1)
                e2s.append(e2)
                rs.append(q.rel.rtype_idx)
        
            e1s = torch.stack(e1s, dim=0).to(args.device)
            e2s = torch.stack(e2s, dim=0).to(args.device)
            if args.use_elmo:
                scores = F.cosine_similarity(e1s, e2s, dim=1)
            else:
                rs = torch.LongTensor(rs).to(args.device)
                remb = model.rel_embeddings(rs)
                
                e1s = model._transfer(e1s, rs)
                e2s = model._transfer(e2s, rs)
                e2s = model._transfer(e2s, rs)
                _scores = model._calc(e1s, e2s, remb)
                scores = torch.sum(_scores, 1)
                if model.norm > 1:
                    scores = torch.pow(scores, 1.0 / model.norm)
            cat_scores[i_cat] = scores
    return cat_scores


def pred_acc(y, scores, threshold):
    if args.use_elmo:
        preds = (scores > threshold).type(torch.int64)
    else:
        preds = (scores < threshold).type(torch.int64)
    return (y == preds).type(torch.float32).sum() / y.shape[0]
    

def dev_thresholds(questions, model, e2idx, embeddings, grid, threshold_min=0.0, threshold_max=100.0):
    best_thresholds = {}
    
    scores = score_questions(questions, model, e2idx, embeddings)
    for i_cat in questions.keys():
        y = torch.LongTensor([label for label, q in questions[i_cat]]).to(args.device)
        best_acc = 0.0
        best_threshold = None
        for t in np.arange(threshold_min, threshold_max, grid):
            acc = pred_acc(y, scores[i_cat], t)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
            logging.debug("cat={}, dev_acc={}, t={}".format(i_cat, acc, t))

        logging.info("cat={}, dev_acc={}, best_threshold={}".format(i_cat, best_acc, best_threshold))
        best_thresholds[i_cat] = best_threshold
    return best_thresholds


def main():
    config = json.load(open(args.training_config, 'r'))
    indices.set_relation_classes(args.relation_config)
    
    if args.use_elmo:
        logging.info("using ELMo")
        elmo = Elmo(args.elmo_option_file, args.elmo_weight_file, 1, dropout=0).to(args.device)
    else:
        pred2idx, idx2pred, _  = indices.load_predicates(config['predicate_indices'])
        argw2idx, idx2argw, _  = indices.load_argw(config['argw_indices'])
        n_preds = len(pred2idx)
        argw_vocabs = argw2idx.keys()
        argw_encoder = create_argw_encoder(config, args.device)
        if args.encoder_file:
            argw_encoder.load(args.encoder_file)
        
        logging.info("model class: " + config['model_type'])
        ModelClass = eval(config['model_type'])
        dnee_model = ModelClass(config, argw_encoder, n_preds, args.device).to(args.device)
        dnee_model.load_state_dict(torch.load(args.model_file,
                                        map_location=lambda storage, location: storage))
    model = elmo if args.use_elmo else dnee_model
    results = torch.zeros((args.n_rounds, len(indices.REL2IDX)), dtype=torch.float32)
    precisions = torch.zeros((args.n_rounds, len(indices.REL2IDX)), dtype=torch.float32)
    recalls = torch.zeros((args.n_rounds, len(indices.REL2IDX)), dtype=torch.float32)
    f1s = torch.zeros((args.n_rounds, len(indices.REL2IDX)), dtype=torch.float32)
    for i_round in range(args.n_rounds):
        logging.info("ROUND {}".format(i_round))

        # dev
        dev_questions = sample_questions(args.dev_question_file, n_cat_questions=500)
        if args.use_elmo:
            dev_e2idx, dev_ev_embeddings = build_elmo(dev_questions, elmo)
        else:
            dev_e2idx, dev_ev_embeddings = build_ev_embeddings(dev_questions, config, pred2idx, argw2idx, dnee_model)
        thresholds = dev_thresholds(dev_questions, model, dev_e2idx, dev_ev_embeddings, args.step_size)

        # test results
        test_questions = sample_questions(args.test_question_file, n_cat_questions=500)
        if args.use_elmo:
            test_e2idx, test_ev_embeddings = build_elmo(test_questions, elmo)
        else:
            test_e2idx, test_ev_embeddings = build_ev_embeddings(test_questions, config, pred2idx, argw2idx, dnee_model)
        test_scores = score_questions(test_questions, model, test_e2idx, test_ev_embeddings)
        for i_cat in test_questions.keys():
            y = torch.LongTensor([label for label, q in test_questions[i_cat]]).to(args.device)
            acc = pred_acc(y, test_scores[i_cat], thresholds[i_cat])

            if args.use_elmo:
                y_preds = (test_scores[i_cat] > thresholds[i_cat]).type(torch.int64)
            else:
                y_preds = (test_scores[i_cat] < thresholds[i_cat]).type(torch.int64)

            y = y.detach().cpu().numpy()
            y_preds = y_preds.detach().cpu().numpy()

            prec = metrics.precision_score(y, y_preds)
            rec = metrics.recall_score(y, y_preds)
            f1 = metrics.f1_score(y, y_preds)
            logging.info("i_cat={} ({}), test_acc={}".format(i_cat, indices.IDX2REL[i_cat], acc))
            logging.info("i_cat={} ({}), test_prec={}".format(i_cat, indices.IDX2REL[i_cat], prec))
            logging.info("i_cat={} ({}), test_rec={}".format(i_cat, indices.IDX2REL[i_cat], rec))
            results[i_round][i_cat] = acc
            precisions[i_round][i_cat] = prec
            recalls[i_round][i_cat] = rec
            f1s[i_round][i_cat] = f1

    avg = torch.mean(results, dim=0)
    avg_precisions = torch.mean(precisions, dim=0)
    avg_recalls = torch.mean(recalls, dim=0)
    avg_f1s = torch.mean(f1s, dim=0)
    for i_cat in test_questions.keys():
        logging.info("i_cat={} ({}), avg_test_acc={} over {} rounds".format(i_cat, indices.IDX2REL[i_cat], avg[i_cat], args.n_rounds))
        logging.info("i_cat={} ({}), avg_test_prec={} over {} rounds".format(i_cat, indices.IDX2REL[i_cat], avg_precisions[i_cat], args.n_rounds))
        logging.info("i_cat={} ({}), avg_test_rec={} over {} rounds".format(i_cat, indices.IDX2REL[i_cat], avg_recalls[i_cat], args.n_rounds))
        logging.info("i_cat={} ({}), avg_test_f1={} over {} rounds".format(i_cat, indices.IDX2REL[i_cat], avg_f1s[i_cat], args.n_rounds))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if args.gpu_id is not None:
        args.device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
