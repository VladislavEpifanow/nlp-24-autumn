import re
from collections import Counter

import pandas as pd

from nltk.util import ngrams
from nltk.collocations import *
from nltk.corpus import stopwords


filter_out_tokens = ['\n', '.', ',', '!', '?', '...', ':', ';']
stop_words = set(stopwords.words('english'))


def get_trigrams(corpus, column='lemma'):
    lemmas = corpus[column]
    trigrams = list(ngrams(lemmas, 3))
    return trigrams

def get_trigrams_freq(trigrams):
    trigrmas_freq = Counter(trigrams)
    return trigrmas_freq

def get_words_count(corpus: pd.DataFrame, column='lemma'):
    return Counter(corpus[column])

def get_number_of_words(data):
    return len(data)

def t_score_trigram(trigrams: dict, words_count: dict, sort=False):
    trigram_t_score = {}

    for trigram in trigrams:
        trigram_freq = trigrams.get(trigram)
        freq_w_1, freq_w_2, freq_w_3 = [words_count.get(word) for word in trigram]
        total_unique_words = len(words_count)

        t_score = (trigram_freq - (freq_w_1*freq_w_2*freq_w_3)/(total_unique_words**(3-1))) / (trigram_freq**0.5)
        trigram_t_score[trigram] = t_score
    
    if sort:
        trigram_t_score = dict(sorted(trigram_t_score.items(), key=lambda item: -item[1]))
    return trigram_t_score

def trigrams_to_pandas(trigrams: dict):
    trigrams_pd = []
    for k, v in trigrams.items():
        trigrams_pd.append([','.join(k), float(v)])
    trigrams_pd = pd.DataFrame(trigrams_pd, columns=['trigrams', 'freq'])
    return trigrams_pd

def read_data_with_filter(filename):

    rows = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] in filter_out_tokens:
                continue
            try:
                token, stem, lemma = line.split('\t')
                lemma = re.sub(r'[^\w\s]','', lemma)
                token = re.sub(r'[^\w\s]','', token)
                if len(lemma) == 0 or token in stop_words:
                    continue
            except Exception as e:
                print(line, e)
                break
            rows.append((token.lower(), stem.lower(),lemma.strip().lower()))
   
    return pd.DataFrame(rows, columns=('token', 'stem', 'lemma'))