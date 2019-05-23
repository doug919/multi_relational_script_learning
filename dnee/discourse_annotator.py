import sys
import os
import logging
import argparse
import json
import time
import re

import progressbar

from dnee import utils


def extract_explicit_connectives(doc, dmarkers):
    sents = doc["sentences"]
    dm_idxs = []
    for i, sent in enumerate(sents):
        toks = [tok["word"].lower() for tok in sent["tokens"]]
        cidx2tidx = {0: 0}
        acc_clen = 0
        for j, tok in enumerate(sent["tokens"]):
            if j == 0:
                acc_clen += (len(tok["word"]) + 1)
                continue
            cidx2tidx[acc_clen] = j
            acc_clen += (len(tok["word"]) + 1)

        text = ' '.join(toks)
        for dtype, dm in dmarkers:
            dlen = len(dm.split(" "))
            tidxs = []
            for m in re.finditer(dm, text):
                if m.start() not in cidx2tidx:
                    continue
                begin_cidx = cidx2tidx[m.start()]
                end_cidx = cidx2tidx[m.start()] + dlen
                tmp_toks = [sents[i]["tokens"][k]["word"].lower() for k in range(begin_cidx, end_cidx)]
                tmp_text = ' '.join(tmp_toks)
                if tmp_text == dm:
                    tidxs.append((i, begin_cidx, end_cidx, dtype, dm))

            dm_idxs += tidxs
    return dm_idxs


def find_min_dist(idxs, conn):
    # get the one close to the connective
    min_dist = 10000
    target_idx = -1
    for idx in idxs:
        dist = abs(idx - conn[1])
        if dist < min_dist:
            min_dist = dist
            target_idx = idx
    return target_idx, min_dist


def find_clause_args(conn, doc, delimiter=';'):
    sent_idx = conn[0]
    sent_toks = [tok['word'] for tok in doc['sentences'][sent_idx]['tokens']]
    idxs = [i for i, tok in enumerate(sent_toks) if tok == delimiter]
    cargs = None
    if len(idxs) > 0:
        idx, dist = find_min_dist(idxs, conn)
        if idx != -1:
            left = (sent_idx, 0, idx)
            right = (sent_idx, idx+1, len(sent_toks))
            cargs = (left, right)
    return cargs


def find_sentence_args(conn, doc):
    sent_idx = conn[0]
    sent_toks = [tok['word'] for tok in doc['sentences'][sent_idx]['tokens']]
    tok_begin_idx = conn[1]
    if len(doc['sentences']) <= 1:
        return None

    arg1, arg2 = None, None
    if sent_idx == 0: # pick right
        arg1 = (sent_idx, 0, len(sent_toks))
        arg2 = (sent_idx+1, 0, len(doc['sentences'][sent_idx+1]))
    elif sent_idx == len(sent_toks) - 1: # pick left
        arg1 = (sent_idx-1, 0, len(doc['sentences'][sent_idx-1]))
        arg2 = (sent_idx, 0, len(sent_toks))
    elif tok_begin_idx <= len(sent_toks) - tok_begin_idx - 1: # pick left
        arg1 = (sent_idx-1, 0, len(doc['sentences'][sent_idx-1]))
        arg2 = (sent_idx, 0, len(sent_toks))
    else: # pick right
        arg1 = (sent_idx-1, 0, len(doc['sentences'][sent_idx-1]))
        arg2 = (sent_idx, 0, len(sent_toks))
    return None if arg1 is None else (arg1, arg2)


def pack_rel(conn, arg1, arg2):
    def _pack_index(k, v):
        ret = {}
        ret[k] = {'sent_idx': v[0], 'tok_begin_idx': v[1], 'tok_end_idx': v[2]}
        return ret

    rel = {}
    rel.update(_pack_index('arg1', arg1))
    rel.update(_pack_index('arg2', arg2))
    rel.update(_pack_index('connective', conn))
    rel['connective']['type'] = conn[3]
    rel['connective']['text'] = conn[4]
    return rel


def annotate(doc, dmarkers):
    if doc is None or len(doc['sentences']) == 0:
        return None

    connectives = extract_explicit_connectives(doc, dmarkers)
    doc_rels = []
    for conn in connectives:
        cargs = find_clause_args(conn, doc, delimiter=';')
        if cargs is not None:
            arg1, arg2 = cargs
            rel = pack_rel(conn, arg1, arg2)
            doc_rels.append(rel)
            continue
        
        cargs = find_clause_args(conn, doc, delimiter=',')
        if cargs is not None:
            arg1, arg2 = cargs
            rel = pack_rel(conn, arg1, arg2)
            doc_rels.append(rel)
            continue

        cargs = find_sentence_args(conn, doc)
        if cargs is None:
            continue
        
        arg1, arg2 = cargs
        rel = pack_rel(conn, arg1, arg2)
        doc_rels.append(rel)
    return doc_rels
