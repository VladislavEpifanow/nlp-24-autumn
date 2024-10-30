import nltk
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, TrigramCollocationFinder
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.util import ngrams
import pandas as pd
from pathlib import Path
import re
from collections import Counter
import math


# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# can I even do this? todo: do this more elegantly
stop_words.update(['educational', 'script', "00000"
                   'draft', 'http', 'purpose', 'module', 'provided', 'spectrum', 'fileviewer',
                   'digcomponents', 'studio', 'studios', 'without', 'marvel', 'inc', '©', '©marvel', 
                   'production', 'duplication', 'dub', 'post', 'revision', 'rev', 'ext'])


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
corpus_length = 0

df = pd.DataFrame()

def read_data_with_filter(filename):

    rows = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            
            try:
                if line == '\n':
                    continue
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


def load_files():
    df = pd.DataFrame(columns=('token', 'stem', 'lemma'))
    frames = [read_data_with_filter(f) for f in Path("../assets/annotated_corpus/train").glob("Action/*.tsv")]
    result = pd.concat(frames)
    return result



def get_trigrams(corpus, column='lemma'):
    text = corpus[column]
    trigrams = list()

    for i in range(len(text) - 2):
        trigrams.append(tuple(text[i:i + 3]))

    # trigrams = list(ngrams(text, 3))
    return trigrams

def get_unigrams(corpus, column='lemma'):
    text = corpus[column]
    trigrams = list()

    for i in range(len(text)):
        trigrams.append(tuple(text[i:i + 1]))

    # trigrams = list(ngrams(text, 3))
    return trigrams

def get_trigrams_count(trigrams):
    return Counter(trigrams)  # why hashes? Because it's faster


def calculate_mutual_information(trigram_counts, unigram_counts, corpus_length):
    mi = 0.0
    result = {}
    total_unigrams = corpus_length
    for (w1, w2, w3), count in trigram_counts.items():
        p_w1_w2_w3 = float(count)/ float(total_unigrams - 2) 
        p_w1 = unigram_counts[(w1,)] / total_unigrams
        p_w2 = unigram_counts[(w2,)] / total_unigrams
        p_w3 = unigram_counts[(w3,)] / total_unigrams
        if p_w1 > 0 and p_w2 > 0 and p_w3 > 0:
            mi =  math.log(p_w1_w2_w3 * corpus_length**2 / (p_w1 * p_w2 * p_w3)) # * p_w1_w2_w3 
            result[(w1, w2, w3)] = mi
    
    return result


def calculate_t_score_trigram(trigram_counts, unigram_counts, corpus_length):
    trigram_t_score = {}

    for trigram in trigram_counts:
        trigram_freq = trigram_counts.get(trigram)
        w_1, w_2, w_3 = [unigram_counts.get((word,)) for word in trigram]
        total_unique_words = corpus_length
        
        # easier to explain with a formula from task
        t_score = (trigram_freq - (w_1*w_2*w_3)/(total_unique_words**(3-1))) / (trigram_freq**0.5)
        trigram_t_score[trigram] = t_score
    
    
    trigram_t_scores = dict(sorted(trigram_t_score.items(), key=lambda item: -item[1]))
    return trigram_t_scores


if __name__ == "__main__":
    corpus = load_files()
    corpus_length = len(corpus['lemma'])
    print(f"Corpus length: {corpus_length}")
    trigrams = get_trigrams(corpus)
    nltk_trigrams = list(ngrams(corpus['lemma'], 3))
    nltk_trigrams_cnt = Counter(nltk_trigrams)

    print("NLTK top 10 trigrams:")
    print(nltk_trigrams_cnt.most_common(10))
    
    print("\n\nTop 10 trigrams:")
    my_trigrams_cnt = get_trigrams_count(trigrams)
    print(my_trigrams_cnt.most_common(10))
    
    print("\n\nNLTK top 10 unigrams:")
    nltk_trigrams = list(ngrams(corpus['lemma'], 1))
    nltk_trigrams_cnt = Counter(nltk_trigrams)
    print(nltk_trigrams_cnt.most_common(10))

    unigrams = get_unigrams(corpus)
    my_unigrams_cnt = get_trigrams_count(unigrams)
    print("\n\n top 10 unigrams:")
    print(my_unigrams_cnt.most_common(10))

    # ground truth
    text = ' '.join(corpus['lemma'].values)
    tokens = nltk.word_tokenize(text, 'english', True)
    nltk_text = nltk.Text(tokens)
    finder_thr = TrigramCollocationFinder.from_words(nltk_text)
    print("\n\n NLTK Mutual information:")
    print(finder_thr.nbest(trigram_measures.mi_like, 20))
    print("\n\n NLTK Score:")
    print(finder_thr.nbest(trigram_measures.student_t, 20))
    

    # my implementation
    result = calculate_mutual_information(my_trigrams_cnt, my_unigrams_cnt, corpus_length=corpus_length)
    print("\n\n Mutual information:")
    print(sorted(result.items(), key=lambda item: -item[1])[:20])


    result = calculate_t_score_trigram(my_trigrams_cnt, my_unigrams_cnt, corpus_length)
    print("\n\n T_scores:")
    print(sorted(result.items(), key=lambda item: -item[1])[:20])
