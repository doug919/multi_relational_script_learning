import re
import sys

import six

from . import indices


class Event(object):
    def __init__(self, pred, arg0, arg0_head, arg1, arg1_head, arg2, arg2_head,
                    sentiment, ani0, ani1, ani2):
        #assert indices.pred2idx is not None
        rep = {'\n': ' ', '::': ' '}
        rep = dict((re.escape(k), v) for k, v in six.iteritems(rep))
        pat = re.compile("|".join(rep.keys()))

        #self.pred = pred if pred in indices.pred2idx else indices.PRED_OOV
        self.pred = pred
        
        # self.pred_idx = indices.pred2idx[self.pred]
        self.arg0 = pat.sub(lambda m: rep[re.escape(m.group(0))], arg0)
        self.arg0_head = arg0_head
        self.arg1 = pat.sub(lambda m: rep[re.escape(m.group(0))], arg1)
        self.arg1_head = arg1_head
        self.arg2 = pat.sub(lambda m: rep[re.escape(m.group(0))], arg2)
        self.arg2_head = arg2_head
        
        if sys.version_info < (3, 0):
            self.pred = self.pred.encode('ascii', 'ignore')
            self.arg0 = self.arg0.encode('ascii', 'ignore')
            self.arg1 = self.arg1.encode('ascii', 'ignore')
            self.arg2 = self.arg2.encode('ascii', 'ignore')
            self.arg0_head = self.arg0_head.encode('ascii', 'ignore')
            self.arg1_head = self.arg1_head.encode('ascii', 'ignore')
            self.arg2_head = self.arg2_head.encode('ascii', 'ignore')
        self.sentiment = sentiment
        # self.sentiment_idx = indices.SENT2IDX[sentiment]
        self.ani0 = ani0
        # self.ani0_idx = indices.ANI2IDX[ani0]
        self.ani1 = ani1
        # self.ani1_idx = indices.ANI2IDX[ani1]
        self.ani2 = ani2
        # self.ani2_idx = indices.ANI2IDX[ani2]

    def __repr__(self):
        return "({}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{})".format(
                self.pred,
                self.arg0, self.arg0_head,
                self.arg1, self.arg1_head,
                self.arg2, self.arg2_head,
                self.sentiment, self.ani0,
                self.ani1, self.ani2)

    def get_pred_index(self, pred2idx):
        return pred2idx[self.pred] if self.pred in pred2idx else pred2idx[indices.PRED_OOV]

    def get_arg_indices(self, argn, argw2idx, use_head=False, arg_len=-1):
        if use_head:
            if argn == 0:
                target = self.arg0_head
            elif argn == 1:
                target = self.arg1_head
            elif argn == 2:
                target = self.arg2_head
            else:
                return None
        else:
            if argn == 0:
                target = self.arg0
            elif argn == 1:
                target = self.arg1
            elif argn == 2:
                target = self.arg2
            else:
                return None
        sp = target.split(' ')
        if arg_len != -1 and len(sp) > arg_len:
            sp = sp[:arg_len]
        # append EOS
        # sp.append(indices.EOS_ARG_WORD)
        ret = [argw2idx[tok] if tok in argw2idx
                else argw2idx[indices.UNKNOWN_ARG_WORD] for tok in sp]
        # padding 0
        if len(ret) < arg_len:
            for i in range(arg_len-len(ret)):
                ret.append(0)
        return ret

    @classmethod
    def from_string(cls, line):
        line = line.rstrip("\n")[1:-1]
        sp = line.split('::')
        obj = cls(sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6],
                    sp[7], sp[8], sp[9], sp[10])
        return obj

    @classmethod
    def from_json(cls, e):
        pred = e['predicate']
        # only use the first sub-argument for now
        arg0_head = e['arg0'][0] if 'arg0' in e else indices.NO_ARG
        arg0 = e['arg0_text'][0] if 'arg0_text' in e else indices.NO_ARG

        arg1_head = e['arg1'][0] if 'arg1' in e else indices.NO_ARG
        arg1 = e['arg1_text'][0] if 'arg1_text' in e else indices.NO_ARG

        arg2_head = e['arg2'][0] if 'arg2' in e else indices.NO_ARG
        arg2 = e['arg2_text'][0] if 'arg2_text' in e else indices.NO_ARG
        sentiment = e['sentiment'] if 'sentiment' in e else None
        ani0 = e['ani0'][0] if 'ani0' in e else indices.UNKNOWN_ANIMACY
        ani1 = e['ani1'][0] if 'ani1' in e else indices.UNKNOWN_ANIMACY
        ani2 = e['ani2'][0] if 'ani2' in e else indices.UNKNOWN_ANIMACY
        obj = cls(pred, arg0, arg0_head, arg1, arg1_head, arg2, arg2_head,
                    sentiment, ani0, ani1, ani2)
        return obj

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__repr__() == other.__repr__()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def valid_pred(self, pred2idx):
        return (self.pred in pred2idx)
    
    def valid_arg_len(self, config):
        # minus one for appending EOS
        if len(self.arg0.split(' ')) > config['arg0_max_len']:
            return False
        if len(self.arg1.split(' ')) > config['arg1_max_len']:
            return False
        if len(self.arg2.split(' ')) > config['arg2_max_len']:
            return False
        return True


class EventRelation:
    def __init__(self, e1, e2, rtype, label=1):
        assert label == 1 or label == 0
        self.label = label
        self.e1 = e1
        self.e2 = e2
        self.rtype = rtype
        self.rtype_idx = indices.REL2IDX[rtype]

    def __repr__(self):
        return "{} ||| {} ||| {}".format(self.rtype_idx,
                                            self.e1,
                                            self.e2)

    @classmethod
    def from_string(cls, line):
        line = line.rstrip("\n")
        sp = line.split(' ||| ')
        assert len(sp) == 3
        rtype_idx = int(sp[0])
        rtype = indices.IDX2REL[rtype_idx]
        e1 = Event.from_string(sp[1])
        e2 = Event.from_string(sp[2])
        obj = cls(e1, e2, rtype)
        return obj

    def is_valid(self, pred2idx, config):
        return self.valid_pred(pred2idx) and self.valid_arg_len(config)

    def valid_pred(self, pred2idx):
        return (self.e1.valid_pred(pred2idx) and self.e2.valid_pred(pred2idx))
    
    def valid_arg_len(self, config):
        return self.e1.valid_arg_len(config) and self.e2.valid_arg_len(config)

    def to_indices(self, pred2idx, argw2idx, use_head=False, arg_len=-1):
        return [self.label,
                self.rtype_idx,
                self.e1.get_pred_index(pred2idx),
                self.e1.get_arg_indices(0, argw2idx, use_head=use_head, arg_len=arg_len),
                self.e1.get_arg_indices(1, argw2idx, use_head=use_head, arg_len=arg_len),
                self.e1.get_arg_indices(2, argw2idx, use_head=use_head, arg_len=arg_len),
                self.e2.get_pred_index(pred2idx),
                self.e2.get_arg_indices(0, argw2idx, use_head=use_head, arg_len=arg_len),
                self.e2.get_arg_indices(1, argw2idx, use_head=use_head, arg_len=arg_len),
                self.e2.get_arg_indices(2, argw2idx, use_head=use_head, arg_len=arg_len)]
