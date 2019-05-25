import sys
import os
import logging
import argparse
import json
import time
import re
import random
import pickle as pkl

import progressbar
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from dnee import utils
from dnee.events import indices
from dnee.datasets.event_relation import EventRelationDataset
from dnee.models import EventTransR, EventTransE, NegativeSampler, ArgWordEncoder, create_argw_encoder
from dnee.models import AttnEventTransE, AttnEventTransR


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='DNEE training')
    parser.add_argument('training_config', metavar='TRAINING_CONFIG',
                        help='config for training')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER',
                        help='output folder for models etc.')

    parser.add_argument('-l', '--prev_best_loss', type=float, default=-1.0,
                        help='previous best loss')
    parser.add_argument('-p', '--prev_folder', default=None,
                        help='previous training outputs')
    parser.add_argument('-e', '--epoch_batch', default=None,
                        help='continued epoch batch numbers, e.g., 1_31111')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-n', '--normalize_embeddings', action='store_true', default=False,
                        help='normalize embeddings after each update')
    parser.add_argument('-r', '--sample_rel', action='store_true', default=False,
                        help='sample relation types for negative sampling')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def save(fld, model, optimizer, i_epoch, i_file, i_batch):
    fpath = os.path.join(fld, "model_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(model.state_dict(), fpath)
    
    fpath = os.path.join(fld, "optim_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(optimizer.state_dict(), fpath)
    
    fpath = os.path.join(fld, "argw_enc_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    model.argw_encoder.save(fpath)


def save_losses(fld, losses):
    fpath = os.path.join(fld, 'losses.pkl')
    pkl.dump(losses, open(fpath, 'wb'))


def main():
    logging.info('using {} for computation.'.format(args.device))
    config = json.load(open(args.training_config, 'r'))

    n_preds = EventTransE.get_n_preds(config['predicate_indices'])
    argw_encoder = create_argw_encoder(config, args.device)
    if args.prev_folder:
        fpath = os.path.join(args.prev_folder, 'argw_enc_{}.pt'.format(args.epoch_batch))
        logging.info('loading {}...'.format(fpath))
        argw_encoder.load(fpath)

    logging.info("model class: " + config['model_type'])
    ModelClass = eval(config['model_type'])
    model = ModelClass(config, argw_encoder, n_preds, args.device).to(args.device)
    if args.prev_folder:
        fpath = os.path.join(args.prev_folder, 'model_{}.pt'.format(args.epoch_batch))
        logging.info('loading {}...'.format(fpath))
        model.load_state_dict(torch.load(fpath, map_location=lambda storage, location: storage))

    sampler = NegativeSampler([config['arg0_max_len'], config['arg1_max_len'], config['arg2_max_len']],
                                config['n_rel_types'],
                                sample_rel=args.sample_rel)

    optimizer = utils.get_optimizer(model, config)
    if args.prev_folder:
        fpath = os.path.join(args.prev_folder, 'optim_{}.pt'.format(args.epoch_batch))
        logging.info('loading {}...'.format(fpath))
        optimizer.load_state_dict(torch.load(fpath, map_location=lambda storage, location: storage))

    arg_lens = [config['arg0_max_len'], config['arg1_max_len']]
    if args.prev_best_loss >= 0:
        min_loss = args.prev_best_loss
        best_epoch = -1
    else:
        min_loss, best_epoch = None, None
    losses = []
    for i_epoch in range(config['n_epochs']):
        files = [os.path.join(config['training_data'], f)
                    for f in os.listdir(config['training_data'])
                    if f.endswith('txt')]
        random.shuffle(files)
        epoch_start = time.time()
        for i_file, f in enumerate(files):
            # ToDo: pre-create these objects and move these outside the loop
            train_data = EventRelationDataset(f, arg_lens)
            train_loader = DataLoader(train_data,
                                        batch_size=config["batch_size"],
                                        shuffle=True,
                                        num_workers=config["n_dataloader_workers"])

            for i_batch, (x, y) in enumerate(train_loader):
                if x.shape[0] != config['batch_size']:
                    logging.debug('batch {} batch_size mismatch, skip.'.format(i_batch))
                    continue

                # sampling
                x_neg = Variable(sampler.sampling(x).to(args.device), requires_grad=False)
                x_pos = Variable(x.to(args.device), requires_grad=False)
                
                model.train()
                p_score = model(x_pos)
                n_score = model(x_neg)
                
                # average over folds of negative samples
                neg_ratio = n_score.shape[0] // p_score.shape[0]
                n_samples = p_score.shape[0]
                for i in range(1, neg_ratio):
                    n_score[:n_samples] = n_score[:n_samples] + n_score[i*n_samples: (i+1)*n_samples]
                n_score = n_score[:n_samples] / neg_ratio
                
                # step
                loss = model.loss_func(p_score, n_score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if args.normalize_embeddings:
                    model.rel_embeddings.weight.data = F.normalize(model.rel_embeddings.weight.data,
                                                                    p=model.norm, dim=1)
                    model.pred_embeddings.weight.data = F.normalize(model.pred_embeddings.weight.data,
                                                                    p=model.norm, dim=1)

                tmp_loss = loss.item()
                losses.append(tmp_loss)
                
                model.eval()
                # dev

                if min_loss:
                    if tmp_loss < min_loss:
                        logging.info("best: {}, {}, {}: loss={}, time={}".format(i_epoch, i_file, i_batch, tmp_loss, time.time()-epoch_start))
                        min_loss = tmp_loss
                        best_epoch = i_epoch
                        save(args.output_folder, model, optimizer, i_epoch, i_file, i_batch)
                        save_losses(args.output_folder, losses)
                else:
                    logging.info("best: {}, {}, {}: loss={}, time={}".format(i_epoch, i_file, i_batch, tmp_loss, time.time()-epoch_start))
                    min_loss = tmp_loss
                    best_epoch = i_epoch
                    save(args.output_folder, model, optimizer, i_epoch, i_file, i_batch)
                    save_losses(args.output_folder, losses)
                
                if i_batch % config['n_batches_per_record'] == 0:
                    logging.info("{}, {}, {}: loss={}, time={}".format(i_epoch, i_file, i_batch, tmp_loss, time.time()-epoch_start))
                    save_losses(args.output_folder, losses)

        logging.info("epoch {}: loss={}, time={}".format(i_epoch, tmp_loss, time.time()-epoch_start))
    
    i_epoch = config["n_epochs"]
    i_file = 0
    i_batch = 0
    save(args.output_folder, model, optimizer, i_epoch, i_file, i_batch)
    save_losses(args.output_folder, losses)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    if torch.cuda.is_available():
        args.device = torch.device('cuda') if args.gpu_id is None \
                else torch.device('cuda:{}'.format(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    main()
