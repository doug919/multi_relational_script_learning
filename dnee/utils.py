import os
import sys
import argparse
import logging
import json

import six
import torch
import torch.optim as optim
import numpy as np
from scipy import spatial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .events import indices, Event, extract_events


DEV_SPLIT = 0
TEST_SPLIT = 1


def micro_f1(y_true, y_pred, n_classes):
    """
        multi class micro F1
    """
    tps, fps, fns = torch.zeros(n_classes, dtype=torch.int32), \
                        torch.zeros(n_classes, dtype=torch.int32), \
                        torch.zeros(n_classes, dtype=torch.int32)
    for i in range(n_classes):
        prediction = (y_pred == i).float()
        truth = (y_true == i).float()
        confusion_vector = prediction / truth
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)

        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        tps[i] = torch.sum(confusion_vector == 1).item()
        fps[i] = torch.sum(confusion_vector == float('inf')).item()
        fns[i] = torch.sum(confusion_vector == 0).item()
    total_tps = tps.sum().float().item()
    total_fps = fps.sum().float().item()
    total_fns = fns.sum().float().item()
    prec = total_tps / (total_tps + total_fps) if total_tps + total_fps != 0.0 else 0.0
    rec = total_tps / (total_tps + total_fns) if total_tps + total_fns != 0.0 else 0.0
    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * prec * rec) / (prec + rec)
    return f1


def parse_text(text, nlp, props):
    parse = None
    server_error = False
    try:
        ann = nlp.annotate(text, properties=props)
    except:
        # server error
        logging.debug("corenlp server error")
        server_error = True

    if not server_error:
        try:
            parse = json.loads(ann)
        except:
            logging.debug("json parse failed; usually timeout")
    return parse


def load_splits(dev_list, test_list):
    def _load(fpath, label):
        splits = {}
        with open(fpath, 'r') as fr:
            for line in fr:
                line = line.rstrip('\n')
                did = '.'.join(line.split('.')[:-1])
                splits[did] = label
        return splits
    
    _dev = _load(dev_list, DEV_SPLIT)
    _test = _load(test_list, TEST_SPLIT)
    _test.update(_dev)
    return _test


def plot_losses(losses, fpath):
    if len(losses) > 0:
        fig = plt.figure()
        plt.title("Loss vs. Batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        x = list(range(len(losses)))
        plt.plot(x, losses, color='red', label='train', linestyle="-")
        plt.savefig(fpath)
        plt.close(fig)


def bin_config(get_arg_func, log_fname=None):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # set logger
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
    try:
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        fpath = os.path.join(args.output_folder, log_fname) if log_fname \
                else os.path.join(args.output_folder, 'log')
    except:
        fpath = log_fname if log_fname else 'log'
    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return args


def load_word_embeddings(fpath, use_torch=False, skip_first_line=False):
    we = {}
    with open(fpath, 'r') as fr:
        for line in fr:
            if skip_first_line:
                skip_first_line = False
                continue
            line = line.rstrip()
            sp = line.split(" ")
            emb = np.squeeze(np.array([sp[1:]], dtype=np.float32))
            we[sp[0]] = torch.from_numpy(emb) if use_torch else emb
    return we


def get_avg_embeddings(words, embeddings):
    dim = embeddings[embeddings.keys()[0]].shape[0]
    total_emb = np.zeros(300, dtype=np.float32)
    cnt = 0
    for w in words:
        if w in embeddings:
            total_emb += embeddings[w]
            cnt += 1

    if cnt == 0:
        emb = np.random.uniform(low=-1.0/dim, high=1.0/dim, size=300)
        total_emb += emb
        cnt += 1
    return total_emb / cnt


def cosine_similarity(v1, v2):
    return 1.0 - spatial.distance.cosine(v1, v2)


def oov_embeddings(dim):
    return np.random.uniform(low=-1.0/dim, high=1.0/dim, size=dim)


def load_jsons(fld_path, func_getid):
    files = [f for f in os.listdir(fld_path) if f.endswith(".json")]
    docs = {}
    for f in files:
        fpath = os.path.join(fld_path, f)
        doc = json.load(open(fpath, 'r'))
        did = func_getid(doc, f)
        docs[did] = doc
    return docs


def find_index(lis, target):
    try:
        idx = lis.index(target)
    except:
        idx = -1
    return idx


def get_optimizer(model, config, **kwargs):
    logging.info("optimizer: {}".format(config['optimizer']))
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    elif config['optimizer'] == 'adagrad':
        if 'lr' in  kwargs:
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
        else:
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()))
    elif config['optimizer'] == 'adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
    elif config['optimizer'] == 'momentum':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    return optimizer


def get_raw_event_repr(e, config, pred2idx, argw2idx, device=None, use_head=False):
    e_len = 1 + config['arg0_max_len'] + config['arg1_max_len']
    raw = torch.zeros(e_len, dtype=torch.int64).to(device) if device else torch.zeros(e_len, dtype=torch.int64)
    pred_idx = e.get_pred_index(pred2idx)
    arg0_idxs = e.get_arg_indices(0, argw2idx, arg_len=config['arg0_max_len'], use_head=use_head)
    arg1_idxs = e.get_arg_indices(1, argw2idx, arg_len=config['arg1_max_len'], use_head=use_head)

    raw[0] = pred_idx
    raw[1: 1+len(arg0_idxs)] = torch.LongTensor(arg0_idxs).to(device) if device else torch.LongTensor(arg0_idxs)
    raw[1+config['arg0_max_len']: 1+config['arg0_max_len']+len(arg1_idxs)] = torch.LongTensor(arg1_idxs).to(device) if device else torch.LongTensor(arg1_idxs)
    return raw


def build_unknown_event():
    return Event(indices.PRED_OOV, indices.NO_ARG, indices.NO_ARG, indices.NO_ARG, indices.NO_ARG,
                        indices.NO_ARG, indices.NO_ARG, None, indices.UNKNOWN_ANIMACY,
                        indices.UNKNOWN_ANIMACY, indices.UNKNOWN_ANIMACY)


def _extract_unique_events(doc, lemmatizer,
        corenlp_dep_key="enhancedPlusPlusDependencies"):
    """
        corenlp_dep_key: collapsed-ccprocessed-dependencies
    """
    events = _extract_events(doc, lemmatizer, corenlp_dep_key=corenlp_dep_key)
    # unique events
    unique_events = {}
    for mid, evs in six.iteritems(events):
        if evs is None:
            continue
        for ev in evs:
            key = '{}_{}_{}_{}'.format(ev['sentidx'],
                    ev['predicate_head_idx'],
                    ev['predicate_head_char_idx_begin'],
                    ev['predicate_head_char_idx_end'])
            if key not in unique_events:
                unique_events[key] = ev
    return list(unique_events.values())


def _extract_events(doc, lemmatizer, corenlp_dep_key="enhancedPlusPlusDependencies"):
    sentences = doc["sentences"]
    doc_corefs = doc["corefs"]
    entities = extract_events.get_all_entities(doc_corefs)

    events = {}
    for coref_key, corefs in six.iteritems(doc_corefs):
        logging.debug("---------------------------")
        logging.debug('coref_key=%s' % coref_key)

        # for each entiy
        for entity in corefs:
            ent_id = entity['id']
            if ent_id in events:
                logging.warning("entity {} has been extracted.".format(ent_id))
                continue

            tmp_events = extract_events.extract_one_multi_faceted_event(
                    sentences, entities,
                    entity, lemmatizer, add_sentence=True,
                    no_sentiment=True,
                    corenlp_dep_key=corenlp_dep_key)
            events[ent_id] = tmp_events
    return events


def save_indices(fpath, idx2item):
    with open(fpath, 'w') as fw:
        for i in range(len(idx2item)):
            fw.write(idx2item[i]+'\n')


def load_indices(fpath):
    item2idx, idx2item = {}, {}
    with open(fpath, 'r') as fr:
        i = 0
        for line in fr:
            line = line.rstrip('\n')
            item2idx[line] = i
            idx2item[i] = line
            i += 1
    return item2idx, idx2item
