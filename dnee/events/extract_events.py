
import os
import sys
import json
import logging
import argparse
from collections import OrderedDict
import time

import six
from nltk.stem.wordnet import WordNetLemmatizer

from . import animacy

# ToDo: we have removed the 'nmod' and 'prep_'. Think abuot if we want them.
# obj_deps = ['obj', 'dobj', 'iobj', 'pobj']
# subj_deps = ['subj', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass']
obj_deps = ['dobj', 'iobj']
subj_deps = ['nsubj', 'nsubjpass']
prep_deps = ['prep_']       # temporally not use
nmod_deps = ['nmod']        # temporally not use
target_deps = obj_deps + subj_deps
target_adj_deps = obj_deps + subj_deps


def get_dep_root(deps):
    for dep in deps:
        if dep['dep'] == 'ROOT':
            dep_root_token = dep['dependentGloss']
            dep_root_tokid = dep['dependent']
            return dep_root_token, dep_root_tokid


def get_ent2verb(tok, tokid, deps):
    cur_tok = tok
    cur_tokid = tokid
    edges = []
    while cur_tok != 'ROOT':
        for dep in deps:
            if dep['dependent'] == cur_tokid:
                assert dep['dependentGloss'] == cur_tok
                edges.append(dep['dep'])
                cur_tok = dep['governorGloss']
                cur_tokid = dep['governor']
                break
    return edges


def find_head_of_toks(start_id, end_id, deps):
    for cur_id in range(start_id, end_id):
        for dep in deps:
            if (cur_id == dep['dependent'] and 
                (dep['governor'] < start_id or dep['governor'] >= end_id)):
                if 'dependentGloss' not in dep:
                    logging.warning('abnormal sentence that has no dependentGloss.')
                    return None, None
                else:
                    return dep['dependentGloss'], dep['dependent']
    return None, None


def find_outgoing_deps(tok, tokid, deps, target_dep_list=None):
    ret_deps = []
    for dep in deps:
        if tokid == dep['governor'] \
            and tok == dep['governorGloss']:

            if target_dep_list is None:
                ret_deps.append(dep)
            else:
                for td in target_dep_list:
                    if dep['dep'].startswith(td):
                        ret_deps.append(dep)
                        break
    return ret_deps


def find_incoming_deps(tok, tokid, deps, target_dep_list=None):
    ret_deps = []
    for dep in deps:
        if tokid == dep['dependent'] \
            and tok == dep['dependentGloss']:

            if target_dep_list is None:
                ret_deps.append(dep)
            else:
                for td in target_dep_list:
                    if dep['dep'].startswith(td):
                        ret_deps.append(dep)
                        break
    return ret_deps


def is_copula(tok, tokid, deps):
    for dep in deps:
        if dep['governor'] == tokid \
            and dep['governorGloss'] == tok \
            and dep['dep'] == 'cop':

            return True
    return False


def is_verb(tok, tokid, toks):
    if tokid == 0:
        assert tok == 'ROOT'
        return False

    assert toks[tokid-1]['index'] == tokid
    assert toks[tokid-1]['word'] == tok
    return toks[tokid-1]['pos'].startswith('VB')


def is_adj(tok, tokid, toks):
    if tokid == 0:
        assert tok == 'ROOT'
        return False

    assert toks[tokid-1]['index'] == tokid
    assert toks[tokid-1]['word'] == tok
    return toks[tokid-1]['pos'].startswith('JJ')


def is_dep_startswith(dep, target_deps):
    for d in target_deps:
        if dep['dep'].startswith(d):
            return True
    return False


def has_outgoing(tok_id, gloss, deps, target_deps):
    for dep in deps:
        if tok_id == dep['governor'] and \
                gloss == dep['governorGloss'] and \
                dep['dep'] in target_deps:
            return dep
    return None


def has_incoming(tok_id, gloss, deps, target_deps):
    for dep in deps:
        if tok_id == dep['dependent'] and \
                gloss == dep['dependentGloss'] and \
                dep['dep'] in target_deps:
            return dep
    return None


def get_negation(key_dep, deps):
    neg = has_outgoing(key_dep['governor'], key_dep['governorGloss'], deps, ['neg'])
    neg_str = ''  if neg is None else 'not_'
    return neg_str


def compose_xcomp(xcomp, deps, lemmatizer):
    ret = ''
    if xcomp is not None:
        # ex: Jim forgot to take it => forget_to_take
        mark = has_outgoing(xcomp['dependent'], xcomp['dependentGloss'], deps, ['mark'])
        ret += lemmatizer.lemmatize(xcomp['governorGloss'], 'v')
        if mark is not None:
            ret += ('_' + mark['dependentGloss'])
        
        prt = has_outgoing(xcomp['dependent'], xcomp['dependentGloss'], deps, ['compound:prt'])
        if prt is not None:
            # ex: Jim took his shirt off => take_off
            ret = ret + '_' + (lemmatizer.lemmatize(prt['governorGloss'], 'v') + '_' + prt['dependentGloss'])
        else:
            ret = ret + '_' + lemmatizer.lemmatize(xcomp['dependentGloss'], 'v')
    return ret


def get_predicate(key_dep, deps, no_comp_pred, no_negation, lemmatizer):
    if no_comp_pred:
        predicate = key_dep['governorGloss']
        if lemmatizer is not None:
            predicate = lemmatizer.lemmatize(predicate, 'v')
    else:
        predicate = ''
        if not no_negation:
            neg_str = get_negation(key_dep, deps)
            predicate += neg_str
        
        prt = has_outgoing(key_dep['governor'], key_dep['governorGloss'], deps, ['compound:prt'])
        if prt is not None:
            # ex: Jim took his shirt off => take_off
            predicate += (lemmatizer.lemmatize(prt['governorGloss'], 'v') + '_' + prt['dependentGloss'])
        else:
            xcomp = has_outgoing(key_dep['governor'], key_dep['governorGloss'], deps, ['xcomp'])
            if xcomp is not None:
                # ex: Jim forgot to take it => forget_to_take
                predicate += compose_xcomp(xcomp, deps, lemmatizer)
                return predicate.lower(), xcomp
            else:
                xcomp_in = has_incoming(key_dep['governor'], key_dep['governorGloss'], deps, ['xcomp'])
                if xcomp_in is not None:
                    predicate += compose_xcomp(xcomp_in, deps, lemmatizer)
                    return predicate.lower(), xcomp_in
                else:
                    predicate += lemmatizer.lemmatize(key_dep['governorGloss'], 'v')
    return predicate.lower(), None


def find_arguments(key_dep, deps, target_deps, except_deps):
    if is_dep_startswith(key_dep, target_deps):
        arg_deps = [key_dep]
    else:
        # search for targets
        arg_deps = find_outgoing_deps(key_dep['governorGloss'], key_dep['governor'], deps, target_deps)
    if except_deps:
        arg_deps = [ad for ad in arg_deps if ad['dep'] not in except_deps]
    return arg_deps


def get_ner(tok, tok_idx, toks):
    w = toks[tok_idx]['word']
    assert w == tok
    return toks[tok_idx]['ner']


def get_entity_by_index(entities, sent_idx, tok_idx):
    for eid, ent in six.iteritems(entities):
        if (sent_idx == ent['sentNum'] - 1
                and tok_idx >= ent['startIndex'] - 1
                and tok_idx < ent['endIndex'] - 1):
            return ent
    return None


def get_arguments(key_dep, xcomp, deps, entities, toks, sent_num, target_deps, except_deps=None):
    arg_deps = find_arguments(key_dep, deps, target_deps, except_deps)
    tmp_arg_texts = []
    tmp_args = []
    tmp_anis = []
    tmp_deps = []
    if len(arg_deps) > 0:
        for ad in arg_deps:
            tmp_args.append(ad['dependentGloss'].lower())
            tmp_deps.append(ad['dep'])
            tok_idx = ad['dependent'] - 1
            tok = ad['dependentGloss']
            ner = get_ner(tok, tok_idx, toks)
            ent = get_entity_by_index(entities, sent_num-1, tok_idx)
            if ent is not None:
                tmp_ani = ent['animacy']
                tmp_anis.append(tmp_ani.lower())
                tmp_arg_texts.append(ent['text'])
            else:
                tmp_ani = animacy.get_animacy(tok, ner)
                tmp_anis.append(tmp_ani.lower())
                tmp_arg_texts.append(ad['dependentGloss'].lower())
    elif xcomp is not None:
        # find target with two verbs inb the xcomp
        if key_dep['governor'] == xcomp['governor']:
            # find xcomp['dependent']
            xcomp_deps = find_outgoing_deps(xcomp['dependentGloss'], xcomp['dependent'], deps, target_deps)
        else:
            # find xcomp['governor']
            xcomp_deps = find_outgoing_deps(xcomp['governorGloss'], xcomp['governor'], deps, target_deps)
        if xcomp_deps is not None:
            for ad in xcomp_deps:
                if except_deps:
                    if ad['dep'] in except_deps:
                        continue

                tmp_args.append(ad['dependentGloss'].lower())
                tmp_deps.append(ad['dep'])
                tok_idx = ad['dependent'] - 1
                tok = ad['dependentGloss']
                ner = get_ner(tok, tok_idx, toks)
                
                ent = get_entity_by_index(entities, sent_num-1, tok_idx)
                if ent is not None:
                    tmp_ani = ent['animacy']
                    tmp_anis.append(tmp_ani.lower())
                    tmp_arg_texts.append(ent['text'])
                else:
                    tmp_ani = animacy.get_animacy(tok, ner)
                    tmp_anis.append(tmp_ani.lower())
                    tmp_arg_texts.append(ad['dependentGloss'].lower())
    return tmp_args, tmp_anis, tmp_deps, tmp_arg_texts


def get_all_entities(doc_corefs):
    entities = {}
    for coref_key, corefs in six.iteritems(doc_corefs):
        # for each entiy
        for entity in corefs:
            ent_id = entity['id']
            if ent_id in entities:
                continue
            entities[ent_id] = entity
    return entities


def get_token_char_pos(toks, pred_head_idx):
    return (toks[pred_head_idx]['characterOffsetBegin'],
            toks[pred_head_idx]['characterOffsetEnd'])


def extract_one_multi_faceted_event(sentences, entities, c_entity, lemmatizer,
                                    add_sentence=True, no_sentiment=False,
                                    corenlp_dep_key="enhancedPlusPlusDependencies"):
    text = c_entity['text']
    sent_num = c_entity['sentNum']
    start_tokid = c_entity['startIndex']
    end_tokid = c_entity['endIndex']
    ent_animacy = c_entity['animacy'] if 'animacy' in c_entity else "UNKNOWN"
    if not no_sentiment:
        sentiment = sentences[sent_num-1]['sentiment']
    logging.debug("entity=%s, sent_num=%d, index=%d:%d+1" % (text, sent_num, start_tokid, end_tokid))

    # get predicates and arguments
    deps = sentences[sent_num-1][corenlp_dep_key]
    toks = sentences[sent_num-1]['tokens']

    # find head token for the entity
    head_tok, head_id = find_head_of_toks(start_tokid, end_tokid, deps)
    if head_tok is None:
        logging.warning('can\'t find head, ignored')
        return None

    # identify target dependencies
    key_deps = find_incoming_deps(head_tok, head_id, deps, target_deps)
    logging.debug('key_deps=%s' % (str(key_deps)))
    
    # ignore incomplete sentence
    if len(key_deps) == 0:
        logging.debug('len(key_deps) is 0')
        return None

    # some times we have multiple key_deps and we want that they appear in the event chain in their token index order.
    if len(key_deps) > 1:
        # sort key_deps by governor's token index
        key_deps = sorted(key_deps, key=lambda k: k["governor"])

    # each entity might have multiple related predicates. We collect them all.
    output_events = []
    for key_dep in key_deps:
        # exclude deps
        #if key_dep['dep'] in exclude_deps:
        #    logging.debug('key_dep in exclude_deps')
        #    continue

        # get compositional predicates
        predicate, xcomp = get_predicate(key_dep, deps, no_comp_pred=False, no_negation=False, lemmatizer=lemmatizer)
        pred_head_idx = key_dep["governor"] - 1
        logging.debug('predicate = %s' % predicate)

        # check POS, we might want adj and verbs
        # ToDo: might not required
        do_create_event = False
        if is_verb(key_dep['governorGloss'], key_dep['governor'], toks):
            do_create_event = True
        elif is_adj(key_dep['governorGloss'], key_dep['governor'], toks):
            if is_dep_startswith(key_dep, target_adj_deps):
                do_create_event = True
        
        # # ignore non-verb
        if do_create_event:
            # create event
            tmp_event = OrderedDict()
            # set all features
            tmp_event['dep'] = key_dep['dep'].lower()
            if add_sentence:
                tmp_event['sentence'] = ' '.join([t['word'] for t in toks])
                tmp_event['sentidx'] = sent_num -1
            tmp_event['predicate'] = predicate
            tmp_event['predicate_head_idx'] = pred_head_idx
            
            pred_head_char_idx_begin, pred_head_char_idx_end = \
                    get_token_char_pos(toks, pred_head_idx)
            tmp_event['predicate_head_char_idx_begin'] = pred_head_char_idx_begin
            tmp_event['predicate_head_char_idx_end'] = pred_head_char_idx_end
            
            if not no_sentiment:
                tmp_event['sentiment'] = sentiment.lower()
            tmp_event['animacy'] = ent_animacy.lower()

            if key_dep['dep'] == 'nsubjpass': # passive tense
                # argument 0 is subj
                tmp_arg0, tmp_ani0, tmp_dep0, tmp_arg0_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['nmod:agent', 'nmod:by'])
                if len(tmp_arg0) > 0:
                    tmp_event['arg0'] = tmp_arg0
                    tmp_event['ani0'] = tmp_ani0
                    tmp_event['dep0'] = tmp_dep0
                    tmp_event['arg0_text'] = tmp_arg0_text

                # argument 1 is dobj
                tmp_arg1, tmp_ani1, tmp_dep1, tmp_arg1_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['nsubjpass'])
                if len(tmp_arg1) > 0:
                    tmp_event['arg1'] = tmp_arg1
                    tmp_event['ani1'] = tmp_ani1
                    tmp_event['dep1'] = tmp_dep1
                    tmp_event['arg1_text'] = tmp_arg1_text
                
                # prep
                tmp_arg2, tmp_ani2, tmp_dep2, tmp_arg2_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['nmod'], ['nmod:agent', 'nmod:by'])
                if len(tmp_arg2) > 0:
                    tmp_event['arg2'] = tmp_arg2
                    tmp_event['ani2'] = tmp_ani2
                    tmp_event['dep2'] = tmp_dep2
                    tmp_event['arg2_text'] = tmp_arg2_text
            else:
                # argument 0 is subj
                tmp_arg0, tmp_ani0, tmp_dep0, tmp_arg0_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['nsubj'])
                if len(tmp_arg0) > 0:
                    tmp_event['arg0'] = tmp_arg0
                    tmp_event['ani0'] = tmp_ani0
                    tmp_event['dep0'] = tmp_dep0
                    tmp_event['arg0_text'] = tmp_arg0_text

                # argument 1 is dobj
                tmp_arg1, tmp_ani1, tmp_dep1, tmp_arg1_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['dobj'])
                if len(tmp_arg1) > 0:
                    tmp_event['arg1'] = tmp_arg1
                    tmp_event['ani1'] = tmp_ani1
                    tmp_event['dep1'] = tmp_dep1
                    tmp_event['arg1_text'] = tmp_arg1_text
                
                # prep
                tmp_arg2, tmp_ani2, tmp_dep2, tmp_arg2_text = \
                        get_arguments(key_dep, xcomp,
                                      deps, entities,
                                      toks, sent_num, ['nmod', 'iobj'])
                if len(tmp_arg2) > 0:
                    tmp_event['arg2'] = tmp_arg2
                    tmp_event['ani2'] = tmp_ani2
                    tmp_event['dep2'] = tmp_dep2
                    tmp_event['arg2_text'] = tmp_arg2_text

            output_events.append(tmp_event)
            logging.debug('!!create event: %s' % (str(tmp_event)))
        else:
            logging.debug('!!drop (%s, %s)' % (key_dep['governorGloss'], key_dep['dep']))
    return output_events


def extract_multi_faceted_events(parsed,
                                    add_sentence=True, no_sentiment=False,
                                    corenlp_dep_key="enhancedPlusPlusDependencies"):
    lemmatizer = WordNetLemmatizer()
    sentences = parsed['sentences']
    doc_corefs = parsed['corefs']
    entities = get_all_entities(doc_corefs)

    entity_events = {}
    for coref_key, corefs in six.iteritems(doc_corefs):
        entity_events[coref_key] = []
        logging.debug("---------------------------")
        logging.debug('coref_key=%s' % coref_key)

        # for each entiy
        for entity in corefs:
            tmp_events = extract_one_multi_faceted_event(sentences, entities,
                                                         entity, lemmatizer, add_sentence,
                                                         no_sentiment, corenlp_dep_key)
            if tmp_events is not None:
                entity_events[coref_key] += tmp_events
    return entity_events


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
    entities = get_all_entities(doc_corefs)

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

            tmp_events = extract_one_multi_faceted_event(
                    sentences, entities,
                    entity, lemmatizer, add_sentence=True,
                    no_sentiment=True,
                    corenlp_dep_key=corenlp_dep_key)
            events[ent_id] = tmp_events
    return events
