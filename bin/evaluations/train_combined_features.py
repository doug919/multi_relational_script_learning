import sys
import os
import logging
import argparse
import json
import time
import re
import random
import pickle as pkl

import h5py
import progressbar
from sklearn.metrics import f1_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
# torch.multiprocessing.set_sharing_strategy('file_system')

from dnee import utils
from dnee.events import indices
from dnee.models import EventTransR, EventTransE, NegativeSampler, ArgWordEncoder, create_argw_encoder
from dnee.models import AttnEventTransR, AttnEventTransE
from dnee.evals import discourse_sense as ds
from dnee.evals.confusion_matrix import Alphabet, ConfusionMatrix
from dnee.models import skipthoughts as st


# elmo_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
# elmo_option_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
elmo_weight_file = "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
elmo_option_file = "elmo_2x2048_256_2048cnn_1xhighway_options.json"


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='training for Discourse Sense')
    parser.add_argument('ds_train_fld', metavar='DS_TRAIN_FLD',
                        help='preprocessed training data')
    parser.add_argument('ds_dev_rel_file', metavar='DS_DEV_REL_FILE',
                        help='gold relation file with events for dev')
    
    parser.add_argument('model_file', metavar='MODEL_FILE',
                        help='modelf file.')
    parser.add_argument('encoder_file', metavar='ENCODER_FILE',
                        help='argument word encoder.')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('relation_config', metavar='RELATION_CONFIG',
                        help='config for relation classes')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER',
                        help='output folder for models etc.')

    parser.add_argument('-s', '--no_dnee_scores', action='store_true', default=False,
                        help='Not using DNEE scores as features')
    parser.add_argument('-n', '--dnee_train_fld', default=None,
                        help='DNEE training data')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('-e', '--n_epoches', type=int, default=5,
                        help='number of epoches')
    parser.add_argument('-r', '--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def save(fld, model, optimizer, i_epoch, i_file, i_batch):
    fpath = os.path.join(ld, "model_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(model.state_dict(), fpath)
    
    fpath = os.path.join(fld, "optim_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(optimizer.state_dict(), fpath)
    
    fpath = os.path.join(fld, "argw_enc_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    model.argw_encoder.save(fpath)


def save_losses(fld, losses):
    fpath = os.path.join(fld, 'losses.pkl')
    pkl.dump(losses, open(fpath, 'w'))


def save_scores(fld, scores, fname):
    fpath = os.path.join(fld, fname)
    pkl.dump(scores, open(fpath, 'w'))


def main():
    logging.info('using {} for computation.'.format(args.device))
    config = json.load(open(args.training_config, 'r'))

    indices.set_relation_classes(args.relation_config)
    pred2idx, idx2pred, _  = indices.load_predicates(config['predicate_indices'])
    argw2idx, idx2argw, _  = indices.load_argw(config['argw_indices'])

    n_preds = len(pred2idx)
    argw_encoder = create_argw_encoder(config, args.device)
    argw_encoder.load(args.encoder_file)
    
    logging.info("model class: " + config['model_type'])
    ModelClass = eval(config['model_type'])
    dnee_model = ModelClass(config, argw_encoder, n_preds, args.device).to(args.device)
    dnee_model.load_state_dict(torch.load(args.model_file,
                                    map_location=lambda storage, location: storage))
    dnee_model.eval()
    
    elmo = Elmo(elmo_option_file, elmo_weight_file, 1, dropout=0)
    train_data = ds.DsDataset(args.ds_train_fld, args.dnee_train_fld)
    
    dev_rels = [json.loads(line) for line in open(args.ds_dev_rel_file)]
    dev_rels = [rel for rel in dev_rels if rel['Type'] != 'Explicit' and rel['Sense'][0] in indices.DISCOURSE_REL2IDX]
    dev_cm, dev_valid_senses = ds.create_cm(dev_rels, indices.DISCOURSE_REL2IDX)

    dnee_seq_len = train_data.dnee_seq_len if args.dnee_train_fld else None
    x_dev, y_dev = ds.get_features(dev_rels, elmo, train_data.seq_len, dnee_model, dnee_seq_len, config, pred2idx, argw2idx, indices.DISCOURSE_REL2IDX,
                                    device=args.device, use_dnee=(args.dnee_train_fld is not None))
    x0_dev, x1_dev, x0_dnee_dev, x1_dnee_dev, x_dnee_dev = x_dev
    x0_dev, x1_dev = x0_dev.to(args.device), x1_dev.to(args.device)

    model = ds.AttentionNN(len(indices.DISCOURSE_REL2IDX), event_dim=config['event_dim'], dropout=args.dropout, use_event=(args.dnee_train_fld is not None), use_dnee_scores=not args.no_dnee_scores).to(args.device)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    logging.info("initial learning rate = {}".format(args.learning_rate))
    logging.info("dropout rate = {}".format(args.dropout))

    # arg_lens = [config['arg0_max_len'], config['arg1_max_len']]
    losses = []
    dev_f1s = []
    best_dev_f1, best_epoch, best_batch = -1, -1, -1
    logging.info("batch_size = {}".format(args.batch_size))
    for i_epoch in range(args.n_epoches):
        train_loader = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=1)

        epoch_start = time.time()
        for i_batch, (x, y) in enumerate(train_loader):
            if y.shape[0] != args.batch_size:
                # skip the last batch
                continue
            if args.dnee_train_fld:
                x0, x1, x0_dnee, x1_dnee, x_dnee = x
                x0 = x0.to(args.device)
                x1 = x1.to(args.device)
                x0_dnee = x0_dnee.to(args.device)
                x1_dnee = x1_dnee.to(args.device)
                x_dnee = x_dnee.to(args.device)
            else:
                x0, x1 = x
                x0 = x0.to(args.device)
                x1 = x1.to(args.device)
                x0_dnee, x1_dnee, x_dnee = None, None, None
            y = y.squeeze().to(args.device)

            model.train()
            optimizer.zero_grad()
            out = model(x0, x1, x0_dnee, x1_dnee, x_dnee)
            loss = model.loss_func(out, y)

            # step
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            model.eval()
            y_pred = model.predict(x0_dev, x1_dev, x0_dnee_dev, x1_dnee_dev, x_dnee_dev)
            dev_prec, dev_recall, dev_f1 = ds.scoring_cm(y_dev, y_pred.cpu(), dev_cm, dev_valid_senses, indices.DISCOURSE_IDX2REL)
            dev_f1s.append(dev_f1)

            ## if i_batch % config['n_batches_per_record'] == 0:
            logging.info("{}, {}: loss={}, time={}".format(i_epoch, i_batch, loss.item(), time.time()-epoch_start))
            logging.info("dev: prec={}, recall={}, f1={}".format(dev_prec, dev_recall, dev_f1))
            if dev_f1 > best_dev_f1:
                logging.info("best dev: prec={}, recall={}, f1={}".format(dev_prec, dev_recall, dev_f1))
                best_dev_f1 = dev_f1
                best_epoch = i_epoch
                best_batch = i_batch
                fpath = os.path.join(args.output_folder, 'best_model.pt')
                torch.save(model.state_dict(), fpath)

    logging.info("{}-{}: best dev f1 = {}".format(best_epoch, best_batch, best_dev_f1))
    fpath = os.path.join(args.output_folder, "losses.pkl")
    pkl.dump(losses, open(fpath, 'wb'))
    fpath = os.path.join(args.output_folder, "dev_f1s.pkl")
    pkl.dump(dev_f1s, open(fpath, 'wb'))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if args.gpu_id:
        args.device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
