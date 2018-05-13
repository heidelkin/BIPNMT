import lib
import functools
import numpy as np

def clean_up_sentence(sent, remove_unk=False, remove_eos=False):
    if lib.Constants.EOS in sent:
        sent = sent[:sent.index(lib.Constants.EOS) + 1]
    if remove_unk:
        sent = filter(lambda x: x != lib.Constants.UNK, sent)
    if remove_eos:
        if len(sent) > 0 and sent[-1] == lib.Constants.EOS:
            sent = sent[:-1]
    return sent


def corpus_bleu(preds, golds):
    assert len(preds) == len(golds)
    clean_preds = []
    clean_golds = []
    for pred, gold in zip(preds, golds):
        pred = clean_up_sentence(pred, remove_unk=False, remove_eos=True)
        gold = clean_up_sentence(gold, remove_unk=False, remove_eos=True)
        clean_preds.append(pred)
        clean_golds.append(gold)
    return lib.Bleu.score_corpus(clean_preds, clean_golds, 4)


class charFEvaluator(object):
    def __init__(self, dict_tgt):
        self.dict_tgt = dict_tgt

    def convertIdxtoWords(self, sentIdx):
        sentIdx = clean_up_sentence(sentIdx, remove_unk=False, remove_eos=False)
        sentWords = [self.dict_tgt.getLabel(w) for w in sentIdx]
        return(sentWords)

    def sentence_charF(self, preds, golds,  phraselevel=False, 
            return_samples=False):
        results = map(functools.partial(self.single_sentence_charF, 
                    phraselevel=phraselevel,
                    return_samples=return_samples), zip(preds, golds))
        if return_samples:
            scores, preds, pred_word, gold_word = zip(*results)
            return(scores, preds, pred_word, gold_word)
        else:
            scores, preds = zip(*results)
            return(scores, preds)


    def single_sentence_charF(self, pair, phraselevel=False, 
            return_samples=False):
        length = len(pair[0])
        pred, gold = pair
        pred = clean_up_sentence(pred, remove_unk=False, remove_eos=False)
        gold = clean_up_sentence(gold, remove_unk=False, remove_eos=False)
        pred_word = [self.dict_tgt.getLabel(w) for w in pred]
        gold_word = [self.dict_tgt.getLabel(w) for w in gold]
        len_pred = len(pred)
        len_gold = len(gold)

        if len_pred == 0:
            score = 0.
            pred = [lib.Constants.PAD] * length
            pred_word = [self.dict_tgt.getLabel(w) for w in pred]
        else:
            gold_input = gold_word[0: len_pred] if phraselevel else gold_word
            score = lib.CharF.score_sentence(pred_word, gold_input, ngrams=6, beta=2.0)
            while len(pred) < length:
                # For batch learning 
                pred.append(lib.Constants.PAD)

        if return_samples:
            return(score, pred, pred_word, gold_word)
        else:
            return(score, pred)



def refine_reward(orig_score, replay_score, pred, replay):
    '''
        Refine part of the sentence level reward    
        by prefix buffer
    '''
    if replay is None:
        return(orig_score)
    replay_cleaned = clean_up_sentence(replay, remove_unk=False, remove_eos=False)
    len_pred = len(pred)

    to_loop = zip(range(len_pred), replay_cleaned[:len_pred]) \
                if len_pred < len(replay_cleaned) else enumerate(replay_cleaned)
    matched = [1 if r == pred[idx] else 0 for idx, r in to_loop] 
    matched = matched + [0] * (len_pred - len(matched))
    score = [replay_score if r == 1 else orig_score for r in matched]
    return(score)
