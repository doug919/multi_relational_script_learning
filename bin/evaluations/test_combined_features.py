"""
    Note that this re-uses some codes from Skip-Thoughts
    https://github.com/ryankiros/skip-thoughts
"""
import os
import argparse
import logging
import time
import json
import codecs
from collections import OrderedDict
import random
from copy import deepcopy

import progressbar
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids

from dnee import utils
from dnee.events import indices, Event, EventRelation
from dnee.models import EventTransR, EventTransE, ArgWordEncoder, create_argw_encoder
from dnee.models import AttnEventTransE, AttnEventTransR
from dnee.events import extract_events
from dnee.evals import discourse_sense as ds


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='evaluate embeddings on Discourse Sense')
    parser.add_argument('ds_dev_file', metavar='DS_DEV_FILE',
                        help='the file for Discourse Sense developing')
    parser.add_argument('ds_test_file', metavar='DS_TEST_FILE',
                        help='the file for Discourse Sense testing')
    parser.add_argument('ds_blind_file', metavar='DS_BLIND_FILE',
                        help='the file for Discourse Sense blind testing')

    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='modelf file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='argument word encoder.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='config for relation classes')
    
    parser.add_argument('ds_model_file', metavar='DS_MODEL_FILE',
                        help='ds modelf file.')
    
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER',
                        help='the folder for outputs.')
    
    parser.add_argument('-w', '--elmo_weight_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
                        help='ELMo weight file')
    parser.add_argument('-p', '--elmo_option_file', default="data/elmo_2x2048_256_2048cnn_1xhighway_options.json",
                        help='ELMo option file')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')

    args = parser.parse_args(argv)
    return args


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


def eval_relations(fpath, fw_path, elmo, dnee_model, config, pred2idx, argw2idx, seq_len, dnee_seq_len):
    rels = [json.loads(line) for line in open(fpath)]
    rels = [rel for rel in rels if rel['Type'] != 'Explicit' and rel['Sense'][0] in indices.DISCOURSE_REL2IDX]
    cm, valid_senses = ds.create_cm(rels, indices.DISCOURSE_REL2IDX)

    x, y = ds.get_features(rels, elmo, seq_len, dnee_model, dnee_seq_len, config, pred2idx, argw2idx, indices.DISCOURSE_REL2IDX,
                            device=args.device, use_dnee=True)
    x0, x1, x0_dnee, x1_dnee, x_dnee = x
    x0, x1 = x0.to(args.device), x1.to(args.device)
    
    model = ds.AttentionNN(len(indices.DISCOURSE_REL2IDX), event_dim=config['event_dim'], use_event=True).to(args.device)
    model.load_state_dict(torch.load(args.ds_model_file, map_location=lambda storage, location: storage))
    model.eval()

    n_rels = len(indices.DISCOURSE_REL2IDX)
    y_pred = model.predict(x0, x1, x0_dnee, x1_dnee, x_dnee)
    
    fw = open(fw_path, 'w')
    for i, rel in enumerate(rels):
        final_pred = y_pred[i].item()
        new_rel = rel_output(rel, indices.DISCOURSE_IDX2REL[final_pred])
        fw.write(json.dumps(new_rel) + '\n')
    fw.close()


def main():
    # DNEE
    t1 = time.time()
    indices.set_relation_classes(args.relation_config)
    config = json.load(open(args.training_config, 'r'))
    pred2idx, idx2pred, _  = indices.load_predicates(config['predicate_indices'])
    argw2idx, idx2argw, _  = indices.load_argw(config['argw_indices'])
    n_preds = len(pred2idx)
    argw_vocabs = argw2idx.keys()
    argw_encoder = create_argw_encoder(config, args.device)
    argw_encoder.load(args.encoder_file)
    
    logging.info("model class: " + config['model_type'])
    ModelClass = eval(config['model_type'])
    dnee_model = ModelClass(config, argw_encoder, n_preds, args.device).to(args.device)
    dnee_model.load_state_dict(torch.load(args.model_file,
                                    map_location=lambda storage, location: storage))
    logging.info('Loading DNEE: {} s'.format(time.time()-t1))
    
    elmo = Elmo(args.elmo_option_file, args.elmo_weight_file, 1, dropout=0)
    
    # DS_TRAIN_FLD = "ds_train_elmo"
    # DNEE_TRAIN_FLD = "ds_train_transe" if config['model_type'] == 'EventTransE' else 'ds_train_transr_tmp'
    # train_data = ds.DsDataset(DS_TRAIN_FLD, DNEE_TRAIN_FLD)
    # logging.info("DNEE_TRAIN_FLD={}".format(DNEE_TRAIN_FLD))
    # seq_len = train_data.seq_len
    # event_seq_len = train_data.dnee_seq_len
    
    # These are the max seq length from training data (above code)
    # We hardcode them to avoid loading training data
    seq_len, event_seq_len = 392, 14
    logging.info("seq_len={}, event_seq_len={}".format(seq_len, event_seq_len))
    
    t1 = time.time()
    logging.info('dev...')
    fw_path = os.path.join(args.output_folder, 'dev_res.json')
    eval_relations(args.ds_dev_file, fw_path, elmo, dnee_model, config, pred2idx, argw2idx, seq_len, event_seq_len)
    logging.info('Eval DEV: {} s'.format(time.time()-t1))
    
    t1 = time.time()
    logging.info('test...')
    fw_path = os.path.join(args.output_folder, 'test_res.json')
    eval_relations(args.ds_test_file, fw_path, elmo, dnee_model, config, pred2idx, argw2idx, seq_len, event_seq_len)
    logging.info('Eval TEST: {} s'.format(time.time()-t1))
    
    t1 = time.time()
    logging.info('blind...')
    fw_path = os.path.join(args.output_folder, 'blind_res.json')
    eval_relations(args.ds_blind_file, fw_path, elmo, dnee_model, config, pred2idx, argw2idx, seq_len, event_seq_len)
    logging.info('Eval BLIND: {} s'.format(time.time()-t1))


if __name__ == '__main__':
    args = utils.bin_config(get_arguments)
    if torch.cuda.is_available():
        args.device = torch.device('cuda') if args.gpu_id is None \
                else torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
