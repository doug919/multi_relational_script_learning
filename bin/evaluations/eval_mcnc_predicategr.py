import sys
import os
import logging
import argparse
import json
import time
import re
import cPickle as pkl

from scipy import sparse
import numpy as np
import torch
import progressbar

from dnee import utils
from dnee.evals import intrinsic
from dnee.events import indices
from dnee.models.predicate_gr import *


# for reproducing the result
np.random.seed(123)


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='MCNC evaluation on PredateGR models')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('question_file', metavar='QUESTION_FILE',
                        help='questions.')
    parser.add_argument('model_class', metavar='MODEL_CLASS', choices=['Word2Vec', 'Word2VecEvent'],
                        help='model class name.')

    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def predict_mcnc(model, q):
    return model.predict_mcnc(q.get_contexts(), q.choices)


def main():
    config = json.load(open(args.training_config, 'r'))
    questions = pkl.load(open(args.question_file, 'r'))
    logging.info("#questions={}".format(len(questions)))
    
    n_correct, n_incorrect = 0, 0
    
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(questions)).start()
    
    ModelClass = eval(args.model_class)
    model = ModelClass(config[args.model_class])
    if 'model_folder' in config[args.model_class]:
        model.load(config[args.model_class]['model_folder'])

    # logging.info("batch_size = {}".format(args.batch_size))
    # batch_size = args.batch_size
    # n_batches = len(questions) // batch_size + 1
    # logging.info("n_batches = {}".format(n_batches))
    i_q = 0
    for q in questions:
        pred = predict_mcnc(model, q)
        if pred == q.ans_idx:
            n_correct += 1
        else:
            n_incorrect += 1
        i_q += 1
        bar.update(i_q)
        logging.debug("pred={}, ans={}".format(pred, q.ans_idx))
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
