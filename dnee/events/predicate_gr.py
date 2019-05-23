"""
    Events in Predicate-GRs form, for competitor models like
        SGNN
        EventComp
        PMI
        SkipGram
"""
import sys
import parse
from ..events import indices


event_parser = parse.compile("{pred}({dep},{arg0},{arg1},{arg2})")



def load_we_index(fpath):
    idx = 1 # save 0 for zero vector
    e2idx = {}
    with open(fpath) as fr:
        for line in fr:
            sp = line.split(' ')
            e2idx[sp[0]] = idx
            idx += 1
    return e2idx


class Event(object):
    def __init__(self, predicate, dep, arg0, arg1, arg2):
        if sys.version_info >= (3, 0):
            self.pred = predicate.lower()
            self.a0 = arg0.lower()
            self.a1 = arg1.lower()
            self.a2 = arg2.lower()
        else:
            self.pred = predicate.encode('ascii', 'ignore').lower()
            self.a0 = arg0.encode('ascii', 'ignore').lower()
            self.a1 = arg1.encode('ascii', 'ignore').lower()
            self.a2 = arg2.encode('ascii', 'ignore').lower()
        self.dep = dep

    def __repr__(self):
        return "{}({},{},{},{})".format(self.pred, self.dep, self.a0, self.a1, self.a2)
    
    @classmethod
    def from_string(cls, line):
        global event_parser
        res = event_parser.parse(line)
        if res is None:
            print(line)
        return cls(res['pred'], res['dep'], res['arg0'], res['arg1'], res['arg2'])
    
    @classmethod
    def from_json(cls, e):
        pred = e['predicate']
        dep = e['dep']
        arg0_head = e['arg0'][0] if 'arg0' in e else indices.NO_ARG
        arg1_head = e['arg1'][0] if 'arg1' in e else indices.NO_ARG
        arg2_head = e['arg2'][0] if 'arg2' in e else indices.NO_ARG
        obj = cls(pred, dep, arg0_head, arg1_head, arg2_head)
        return obj

    def index(self, pred2idx, argw2idx):
        if self.pred not in pred2idx:
            return None
        idxs = [pred2idx[self.pred]]
        for a in [self.a0, self.a1, self.a2]:
            if a in argw2idx:
                idxs.append(argw2idx[a])
            else:
                idxs.append(argw2idx[indices.UNKNOWN_ARG_WORD])
        import pdb; pdb.set_trace()
        return idxs

    @staticmethod
    def cj08_format(pred, dep):
        return '({},{})'.format(pred, dep)

    @staticmethod
    def predicategr_format(pred, dep, a0, a1, a2, protagonist_str='_PROTAGONIST_'):
        if dep == 'nsubj':
            estr = '{}({},{},{})'.format(pred, protagonist_str, a1, a2)
        elif dep == 'dobj' or dep == 'nsubjpass':
            estr = '{}({},{},{})'.format(pred, a0, protagonist_str, a2)
        else:
            estr = '{}({},{},{})'.format(pred, a0, a1, protagonist_str)
        return estr

    def to_cj08_format(self):
        return self.cj08_format(self.pred, self.dep)
    
    def to_predicategr_format(self):
        return self.predicategr_format(self.pred, self.dep, self.a0, self.a1, self.a2)


class EventChain(object):
    def __init__(self, events):
        for e in events:
            assert isinstance(e, Event)
        self.events = events
    
    def __repr__(self):
        es = [repr(e) for e in self.events]
        return ' '.join(es)

    def gen(self):
        for e in self.events:
            yield e

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx]

    @classmethod
    def from_string(cls, line):
        line = line.rstrip('\n')
        sp = line.split(' ')
        events = [Event.from_string(e) for e in sp]
        return cls(events)
