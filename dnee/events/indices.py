import json
from collections import OrderedDict

import six


# animacy
UNKNOWN_ANIMACY = "unknown"
ANI2IDX= {UNKNOWN_ANIMACY: 0, "animate": 1, "inanimate": 2}
IDX2ANI = {0: UNKNOWN_ANIMACY, 1: "animate", 2: "inanimate"}

# sentiment
SENT2IDX = {"verynegative":0,
        "negative": 1,
        "neutral": 2,
        "positive": 3,
        "verypositive": 4}
IDX2SENT = {0: "verynegative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "verypositive"}

# relations
REL_COREF = "Coref"
REL_CONTEXT = "Context"
REL2IDX = {}
# REL2IDX = {"Comparison.Contrast": 0,
        # "Contingency.Cause.Reason": 1,
        # "Contingency.Cause.Result": 2,
        # "Contingency.Condition": 3,
        # "Expansion.Restatement": 4,
        # "Expansion.Conjunction": 5,
        # "Expansion.Instantiation": 6,
        # "Temporal.Synchrony": 7,
        # "Temporal.Asynchronous": 8,
        # REL_CONTEXT: 9,
        # REL_COREF: 10
        # }

DISCOURSE_REL2IDX = {}
DISCOURSE_IDX2REL = {}
# DISCOURSE_REL2IDX = {"Comparison.Contrast": 0,
        # "Contingency.Cause.Reason": 1,
        # "Contingency.Cause.Result": 2,
        # "Contingency.Condition": 3,
        # "Expansion.Restatement": 4,
        # "Expansion.Conjunction": 5,
        # "Expansion.Instantiation": 6,
        # "Temporal.Synchrony": 7,
        # "Temporal.Asynchronous": 8
        # }

IDX2REL = {}
# IDX2REL = {0: "Comparison.Contrast",
        # 1: "Contingency.Cause.Reason",
        # 2: "Contingency.Cause.Result",
        # 3: "Contingency.Condition",
        # 4: "Expansion.Restatement",
        # 5: "Expansion.Conjunction",
        # 6: "Expansion.Instantiation",
        # 7: "Temporal.Synchrony",
        # 8: "Temporal.Asynchronous",
        # 9: REL_CONTEXT,
        # 10: REL_COREF
        # }

# constants
PRED_OOV = '__PRED_OOV__'
NO_ARG = '__NO_ARG__'
UNKNOWN_ARG_WORD = 'UNK'
EOS_ARG_WORD = '<eos>'


def set_relation_classes(fpath):
    global REL2IDX, IDX2REL, DISCOURSE_REL2IDX
    rel_config = json.load(open(fpath, 'r'))
    REL2IDX = rel_config['rel2idx']
    IDX2REL = {v: k for k, v in six.iteritems(REL2IDX)}

    db, de = int(rel_config['disc_begin']), int(rel_config['disc_end'])
    for i in range(db, de):
        DISCOURSE_REL2IDX[IDX2REL[i]] = i
        DISCOURSE_IDX2REL[i] = IDX2REL[i]


def load_freqs(fpath):
    ret = {}
    with open(fpath) as fr:
        for line in fr:
            line = line.rstrip('\n')
            sp = line.split('\t')
            ret[sp[0]] = int(sp[1])
    return ret


def dump_freqs(output_file, freq):
    with open(output_file, 'w') as fw:
        if isinstance(freq, dict):
            for k, v in six.iteritems(freq):
                fw.write('{}\t{}\n'.format(k, v))
        else:
            for p in freq:
                fw.write('{}\t{}\n'.format(p[0], p[1]))


def load_predicates(fpath):
    return _load_freqs(fpath, oov_key=PRED_OOV)


def load_argw(fpath, eos_key=EOS_ARG_WORD):
    return _load_freqs(fpath, oov_key=UNKNOWN_ARG_WORD,
                        eos_key=eos_key, begin_idx=1)


def _load_freqs(fpath, oov_key=None, eos_key=None, begin_idx=0):
    key2idx, idx2key = OrderedDict(), OrderedDict()
    key_freq = {}
    idx = begin_idx
    if oov_key:
        key2idx[oov_key] = idx
        idx2key[idx] = oov_key
        idx += 1
    if eos_key:
        key2idx[eos_key] = idx
        idx2key[idx] = eos_key
        idx += 1
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.rstrip('\n')
            sp = line.split('\t')
            if sp[0] not in key2idx:
                key2idx[sp[0]] = idx
                idx2key[idx] = sp[0]
                idx += 1
            key_freq[sp[0]] = int(sp[1])
    return key2idx, idx2key, key_freq


def load_dictionary(fpath):
    with open(fpath, 'r') as fr:
        all_lines = fr.readlines()
    argw2idx = {word.strip(): i for i, word in enumerate(all_lines)}
    return argw2idx
