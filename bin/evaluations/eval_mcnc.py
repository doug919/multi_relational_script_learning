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

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder, AttnEventTransE, AttnEventTransR


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='MCNC evaluation')
    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='model file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='encoder file.')
    parser.add_argument('question_file', metavar='QUESTION_FILE',
                        help='questions.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='config for relations')

    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='batch size for evaluation')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-c', '--context_rel', action='store_true', default=False,
                        help='use REL_CONTEXT instead of REL_COREF')
    parser.add_argument('-u', '--use_head', action='store_true', default=False,
                        help='use head word only for arguments')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def build_embeddings(model, questions, config, pred2idx, argw2idx, rtype):
    e2idx = {}
    idx = 0
    for q in questions:
        for ctx in q.get_contexts():
            key = ctx.__repr__()
            if key not in e2idx:
                e2idx[key] = idx
                idx += 1

        for ch in q.choices:
            key = ch.__repr__()
            if key not in e2idx:
                e2idx[key] = idx
                idx += 1
    e_len = 1 + config['arg0_max_len'] + config['arg1_max_len']
    inputs = torch.zeros((len(e2idx), e_len),
                    dtype=torch.int64).to(args.device)

    for q in questions:
        for e in q.get_contexts():
            idx = e2idx[e.__repr__()]
            inputs[idx] = utils.get_raw_event_repr(e, config, pred2idx, argw2idx, device=args.device, use_head=args.use_head)
        for e in q.choices:
            idx = e2idx[e.__repr__()]
            inputs[idx] = utils.get_raw_event_repr(e, config, pred2idx, argw2idx, device=args.device, use_head=args.use_head)
    embeddings = model._transfer(model.embed_event(inputs), rtype)
    return e2idx, embeddings


def main():
    config = json.load(open(args.training_config, 'r'))
    indices.set_relation_classes(args.relation_config)
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
    
    n_correct, n_incorrect = 0, 0
    rtype = indices.REL2IDX[indices.REL_CONTEXT] if args.context_rel else indices.REL2IDX[indices.REL_COREF]
    rtype = torch.LongTensor([rtype]).to(args.device)
    
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(questions)).start()
    
    logging.info("batch_size = {}".format(args.batch_size))
    batch_size = args.batch_size
    n_batches = len(questions) // batch_size + 1
    logging.info("n_batches = {}".format(n_batches))
    i_q = 0
    for i_batch in range(n_batches):
        batch_questions = questions[i_batch*batch_size: (i_batch+1)*batch_size]
        e2idx, embeddings = build_embeddings(model, batch_questions, config, pred2idx, argw2idx, rtype)
        for q in batch_questions:
            pred = intrinsic.predict_mcnc(model, q, e2idx, embeddings, rtype, args.device)
            if pred == q.ans_idx:
                n_correct += 1
            else:
                n_incorrect += 1
            i_q += 1
            bar.update(i_q)
    bar.finish()
    print("n_correct={}, n_incorrect={}".format(n_correct, n_incorrect))
    print("accuracy={}".format(float(n_correct)/(n_correct+n_incorrect)))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if torch.cuda.is_available():
        args.device = torch.device('cuda') if args.gpu_id is None \
                else torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
