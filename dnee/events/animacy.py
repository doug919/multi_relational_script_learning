"""Module for determining animacy

    Last Update: July 15th 2016
    Author: I-Ta Lee @Purdue
"""
import os
import sys
from io import open


# ToDo: create module configurations
module_base = os.path.dirname(__file__)
animate_words_fpath = os.path.join(module_base, 'animacy_data/animate.unigrams.txt')
inanimate_words_fpath = os.path.join(module_base, 'animacy_data/inanimate.unigrams.txt')
animate_words = None
inanimate_words = None

ANIMATE_STR = 'ANIMATE'
INANIMATE_STR = 'INANIMATE'
UNKNOWN_ANIMACY_STR = 'UNKNOWN'

animate_pronouns = ["i", "me", "myself", "mine", "my", "we", "us", "ourself", "ourselves", "ours", "our", "you", "yourself", "yours", "your", "yourselves", "he", "him", "himself", "his", "she", "her", "herself", "hers", "her", "one", "oneself", "one's", "they", "them", "themself", "themselves", "theirs", "their", "they", "them", "'em", "themselves", "who", "whom", "whose"]
inanimate_pronouns = ["it", "itself", "its", "where", "when"]


def get_animacy_by_index(entities, sent_idx, tok_idx):
    for eid, ent in entities.iteritems():
        if (sent_idx == ent['sentNum'] - 1
                and tok_idx >= ent['startIndex'] - 1
                and tok_idx < ent['endIndex'] - 1):
            return ent['animacy']
    return None


def get_animacy_from_corefs(corefs, sent_idx, tok_idx):
    for cid, cchain in corefs.iteritems():
        for ent in cchain:
            if (sent_idx == ent['sentNum'] - 1
                    and tok_idx >= ent['startIndex'] - 1
                    and tok_idx < ent['endIndex'] - 1):
                return ent['animacy']
    return None


def get_animacy_from_parsed(parsed, doc_id, sent_idx, tok_idx):
    corefs = parsed[doc_id]['corefs']
    return get_animacy_from_corefs(corefs, sent_idx, tok_idx)


def load_animacy_file(fpath):
    """load animacy list file

    Args:
        fpath: File path

    Returns:
        A list of words.

    """
    ret_list = []
    with open(fpath, encoding='utf-8') as fr:
        for line in fr:
            ret_list.append(line.rstrip('\n'))
    return ret_list


def get_animacy_auto(word, ner, entities, sent_idx, tok_idx):
    tmp_ani = get_animacy_by_index(entities, sent_idx, tok_idx)
    if tmp_ani is None:
        tmp_ani = get_animacy(word, ner)
    return tmp_ani


def get_animacy(word, ner):
    """Get animacy (animate or inanimate) for a word

    Args:
        word: The word to be checked. It should be a word without any space.
        ner: Named entity recognition label of the word.

    Returns:
        A string 'ANIMATE' or 'INANIMATE'.

    """

    ret_animacy = UNKNOWN_ANIMACY_STR
    # determine using NER, pronoun list, and word list
    if ner == "PERSON" or ner.startswith('PER'):
        ret_animacy = ANIMATE_STR
    elif (ner == 'LOCATION'
            or ner.startswith('LOC')
            or ner == 'MONEY'
            or ner == 'NUMBER'
            or ner == 'PERCENT'
            or ner ==  'DATE'
            or ner == 'TIME'
            or ner.startswith('FAC')
            or ner.startswith('GPE')
            or ner.startswith('WEA')
            or ner.startswith('ORG')):
        ret_animacy = INANIMATE_STR
    elif word in animate_pronouns:      # check with pronoun list
        ret_animacy = ANIMATE_STR
    elif word in inanimate_pronouns:    # check with pronoun list
        ret_animacy = INANIMATE_STR
    else:
        # check with animate/inanimate words
        global animate_words
        global animate_words_fpath
        global inanimate_words
        global inanimate_words_fpath
        if animate_words is None:
            animate_words = load_animacy_file(animate_words_fpath)
        if inanimate_words is None:
            inanimate_words = load_animacy_file(inanimate_words_fpath)
        if word in animate_words:
            ret_animacy = ANIMATE_STR
        elif word in inanimate_words:
            ret_animacy = INANIMATE_STR
    return ret_animacy
