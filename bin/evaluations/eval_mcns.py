import sys
import os
import logging
import argparse
import json
import time
import re
import pickle as pkl
import random

import numpy as np
import torch
import progressbar

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder
from dnee.models import AttnEventTransE, AttnEventTransR

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='MCNS and MCNE evaluation')
    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='model file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='encoder file.')
    parser.add_argument('word_embedding_file', metavar='WE_FILE',
                        help='the file of pre-trained word embeddings')
    parser.add_argument('question_file', metavar='QUESTION_FILE',
                        help='questions.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='config for relations')

    parser.add_argument('inference_model', metavar='INFERENCE_MODEL', choices=['Viterbi', 'Baseline', 'Skyline'],
                        help='Model to conduct inferece {"Viterbi", "Baseline", "Skyline"}.')

    parser.add_argument('-s', '--no_subsample', action='store_true', default=False,
                        help='do not subsample the questions')
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


def event_to_we(e, we, dim):
    toks = e.pred.split('_')
    cnt = 0
    emb = torch.zeros(dim, dtype=torch.float32)
    for tok in toks:
        if tok not in we:
            continue
        else:
            emb += we[tok]
            cnt += 1
    if cnt > 0:
        emb = emb / cnt
    else:
        emb = np.random.uniform(low=-1.0/dim, high=1.0/dim, size=dim)
        emb = torch.from_numpy(emb)
    return emb


def build_embeddings(model, questions, config, pred2idx, argw2idx, rtype, we):
    e2idx = {}
    idx = 0
    for q in questions:
        key = q.echain[0].__repr__()
        if key not in e2idx:
            e2idx[key] = idx
            idx += 1

        for clist in q.choice_lists:
            for c in clist:
                key = c.__repr__()
                if key not in e2idx:
                    e2idx[key] = idx
                    idx += 1
    e_len = 1 + config['arg0_max_len'] + config['arg1_max_len']
    inputs = torch.zeros((len(e2idx), e_len),
                    dtype=torch.int64).to(args.device)
    wdim = we[we.keys()[0]].shape[0]
    w_embeddings = torch.zeros((len(e2idx), wdim),
                    dtype=torch.float32).to(args.device)

    for q in questions:
        key = q.echain[0].__repr__()
        idx = e2idx[key]
        inputs[idx] = utils.get_raw_event_repr(q.echain[0], config, pred2idx, argw2idx, device=args.device, use_head=args.use_head)
        w_embeddings[idx] = event_to_we(q.echain[0], we, wdim)

        for clist in q.choice_lists:
            for c in clist:
                key = c.__repr__()
                idx = e2idx[key]
                inputs[idx] = utils.get_raw_event_repr(c, config, pred2idx, argw2idx, device=args.device, use_head=args.use_head)
                w_embeddings[idx] = event_to_we(c, we, wdim)
    ev_embeddings = model._transfer(model.embed_event(inputs), rtype)
    return e2idx, ev_embeddings, w_embeddings


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
    
    we = utils.load_word_embeddings(args.word_embedding_file, use_torch=True)
    
    questions = pkl.load(open(args.question_file, 'r'))
    if not args.no_subsample:
        # ridxs = list(range(len(questions)))
        # random.shuffle(ridxs)
        # ridxs = [ridxs[i] for i in range(1000)]
        # questions = [questions[i] for i in ridxs]
        questions = questions[:10000]
    logging.info("#questions={}".format(len(questions)))
    
    rtype = indices.REL2IDX[indices.REL_CONTEXT] if args.context_rel else indices.REL2IDX[indices.REL_COREF]
    rtype = torch.LongTensor([rtype]).to(args.device)
    
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(questions)).start()
    
    logging.info("batch_size = {}".format(args.batch_size))
    batch_size = args.batch_size
    n_batches = len(questions) // batch_size + 1 if len(questions) % batch_size != 0 else len(questions) // batch_size
    logging.info("#questions = {}".format(len(questions)))
    logging.info("n_batches = {}".format(n_batches))
    i_q = 0

    # ((we, ev, mix), (mcns, mcne),(incorrect, correct))
    WE_IDX, EV_IDX, MIX_IDX = 0, 1, 2
    MCNS_IDX, MCNE_IDX = 0, 1
    INCORRECT_IDX, CORRECT_IDX = 0, 1
    results = torch.zeros((3, 2, 2), dtype=torch.int64)
    for i_batch in range(n_batches):
        batch_questions = questions[i_batch*batch_size: (i_batch+1)*batch_size]
        e2idx, ev_embeddings, w_embeddings = build_embeddings(model, batch_questions, config, pred2idx, argw2idx, rtype, we)
        for q in batch_questions:
            # when calculating the accuracy, we only consider the questions in the middle
            # so that MCNS and MCNE can have a fair comparison
            n_q = len(q.ans_idxs) - 1

            we_preds, ev_preds = intrinsic.predict_mcns(model, q, e2idx, ev_embeddings, w_embeddings, rtype, args.device, args.inference_model)
            for i in range(n_q):
                for emb_idx, preds in [(WE_IDX, we_preds), (EV_IDX, ev_preds)]:
                    if preds[i] == q.ans_idxs[i]:
                        results[emb_idx][MCNS_IDX][CORRECT_IDX] += 1
                    else:
                        results[emb_idx][MCNS_IDX][INCORRECT_IDX] += 1
            
            we_preds, ev_preds = intrinsic.predict_mcne(model, q, e2idx, ev_embeddings, w_embeddings, rtype, args.device, args.inference_model)
            for i in range(n_q):
                for emb_idx, preds in [(WE_IDX, we_preds), (EV_IDX, ev_preds)]:
                    if preds[i] == q.ans_idxs[i]:
                        results[emb_idx][MCNE_IDX][CORRECT_IDX] += 1
                    else:
                        results[emb_idx][MCNE_IDX][INCORRECT_IDX] += 1
            i_q += 1
            bar.update(i_q)
    bar.finish()

    results = results.type(torch.float32)
    print ("MCNS:")
    print ("\tWE:")
    print("\t\taccuracy={}".format(results[WE_IDX][MCNS_IDX][CORRECT_IDX]/(results[WE_IDX][MCNS_IDX][CORRECT_IDX]+results[WE_IDX][MCNS_IDX][INCORRECT_IDX])))
    print ("\tEV:")
    print("\t\taccuracy={}".format(results[EV_IDX][MCNS_IDX][CORRECT_IDX]/(results[EV_IDX][MCNS_IDX][CORRECT_IDX]+results[EV_IDX][MCNS_IDX][INCORRECT_IDX])))
    # print ("\tMIX:")
    # print("\t\taccuracy={}".format(results[MIX_IDX][MCNS_IDX][CORRECT_IDX]/(results[MIX_IDX][MCNS_IDX][CORRECT_IDX]+results[MIX_IDX][MCNS_IDX][INCORRECT_IDX])))
    
    print ("MCNE:")
    print ("\tWE:")
    print("\t\taccuracy={}".format(results[WE_IDX][MCNE_IDX][CORRECT_IDX]/(results[WE_IDX][MCNE_IDX][CORRECT_IDX]+results[WE_IDX][MCNE_IDX][INCORRECT_IDX])))
    print ("\tEV:")
    print("\t\taccuracy={}".format(results[EV_IDX][MCNE_IDX][CORRECT_IDX]/(results[EV_IDX][MCNE_IDX][CORRECT_IDX]+results[EV_IDX][MCNE_IDX][INCORRECT_IDX])))
    # print ("\tMIX:")
    # print("\t\taccuracy={}".format(results[MIX_IDX][MCNE_IDX][CORRECT_IDX]/(results[MIX_IDX][MCNE_IDX][CORRECT_IDX]+results[MIX_IDX][MCNE_IDX][INCORRECT_IDX])))

    logging.info("MCNS:")
    logging.info("\tWE:")
    logging.info("\t\taccuracy={}".format(results[WE_IDX][MCNS_IDX][CORRECT_IDX]/(results[WE_IDX][MCNS_IDX][CORRECT_IDX]+results[WE_IDX][MCNS_IDX][INCORRECT_IDX])))
    logging.info("\tEV:")
    logging.info("\t\taccuracy={}".format(results[EV_IDX][MCNS_IDX][CORRECT_IDX]/(results[EV_IDX][MCNS_IDX][CORRECT_IDX]+results[EV_IDX][MCNS_IDX][INCORRECT_IDX])))
    # logging.info("\tMIX:")
    # logging.info("\t\taccuracy={}".format(results[MIX_IDX][MCNS_IDX][CORRECT_IDX]/(results[MIX_IDX][MCNS_IDX][CORRECT_IDX]+results[MIX_IDX][MCNS_IDX][INCORRECT_IDX])))
    
    logging.info("MCNE:")
    logging.info("\tWE:")
    logging.info("\t\taccuracy={}".format(results[WE_IDX][MCNE_IDX][CORRECT_IDX]/(results[WE_IDX][MCNE_IDX][CORRECT_IDX]+results[WE_IDX][MCNE_IDX][INCORRECT_IDX])))
    logging.info("\tEV:")
    logging.info("\t\taccuracy={}".format(results[EV_IDX][MCNE_IDX][CORRECT_IDX]/(results[EV_IDX][MCNE_IDX][CORRECT_IDX]+results[EV_IDX][MCNE_IDX][INCORRECT_IDX])))
    # logging.info("\tMIX:")
    # logging.info("\t\taccuracy={}".format(results[MIX_IDX][MCNE_IDX][CORRECT_IDX]/(results[MIX_IDX][MCNE_IDX][CORRECT_IDX]+results[MIX_IDX][MCNE_IDX][INCORRECT_IDX])))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if torch.cuda.is_available():
        args.device = torch.device('cuda') if args.gpu_id is None \
                else torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
