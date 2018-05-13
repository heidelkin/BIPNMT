import numpy as np
from collections import defaultdict 


def get_ngrams(sent_char, ngrams):
    """
        function to update the ngram count for the input sentence
        args:
        - sent_char: sentence tokenzied in character level
    """
    ngrams_dict = defaultdict(int)
    length = len(sent_char)

    for n in range(1, ngrams + 1):
        for i in range(length - n + 1):
            ngram = tuple(sent_char[i : (i + n)])
            ngrams_dict[ngram] += 1
    return(ngrams_dict, length)


def score_sentence(hypo, ref, ngrams, beta):
    chrP_avg, chrR_avg = chrP_chrR(hypo, ref, ngrams)
    if (not (chrP_avg <= 1e-16)) and (not (chrR_avg <= 1e-16)):
        return((1 + beta**2) * (chrP_avg * chrR_avg) / ((chrP_avg * (beta**2) + chrR_avg)))
    else:
        return(0.0)

def chrP_chrR(hypo, ref, ngrams):
    """
        function to compute char-ngram (precision and recall) which 
        is averaged over all ngrams for each input pair of hypo, ref
    """

    # Case checking: empty string:
    if (len(hypo) < 1) or (len(ref) < 1):
        if hypo == ref:
            return(1.0, 1.0) 
        else:
            return(0.0, 1e-16)

    # Break down the word tokens to char tokens
    char_hyp = " ".join(hypo)
    char_ref = " ".join(ref)

    # Dict to store all character ngrams 
    hyp_ngrams, length_hyp = get_ngrams(char_hyp, ngrams)
    ref_ngrams, length_ref = get_ngrams(char_ref, ngrams)

    # p: precision; p[1][0]: overlapped uni-gram; 
    # p[1][1]; total unigram hyp
    p = []
    r = []
    for n in range(ngrams + 1):
        p.append([0, 0])
        r.append([0, 0])

    for hyp_key, hyp_val in hyp_ngrams.items():
        n = len(hyp_key)
        p[n][0] += min(hyp_val, ref_ngrams[hyp_key]) 
        p[n][1] += hyp_val

    for ref_key, ref_val in ref_ngrams.items():
        n = len(ref_key)
        r[n][0] += min(ref_val, hyp_ngrams[ref_key]) 
        r[n][1] += ref_val

    # Average the precision and recall over n-grams
    ngr_p = 0.0
    ngr_r = 0.0 
    for ngram in range(1, ngrams + 1):
        # 0 len for translation in that ngram
        if p[ngram][1] == 0:
            ngr_p += 0
        else:
            ngr_p += p[ngram][0] / p[ngram][1]
        if r[ngram][1] == 0:
            ngr_r += 0
        else:
            ngr_r += r[ngram][0] / r[ngram][1]
    
    if ((length_hyp >= ngrams) and (length_ref >= ngrams)):
        normalizer = ngrams
    else:
        if ((length_hyp < ngrams) and (length_ref >= ngrams)):
            normalizer = length_hyp
        elif ((length_hyp >= ngrams) and (length_ref < ngrams)):
            normalizer = length_ref
        else:
            # when both < ngrams
            normalizer = length_hyp

    ngr_p /= normalizer
    ngr_r /= normalizer
    return(ngr_p, ngr_r)
       
