import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')


from nltk.collocations import *

from n_gram import *
from utils import *

def main():

    data = read_data_with_filter('projects\\danishevskiy-lab\\assets\\annotated-corpus\\results\\big_tokens.tsv')

    trigrams = get_trigrams(data, 'lemma')
    trigrams_freq = get_trigrams_freq(trigrams)

    words_count = get_words_count(data, 'lemma')
    t_score_trigrams = t_score_trigram(trigrams_freq, words_count, sort=True)
    
    # log trigrams
    trigrams_pd = trigrams_to_pandas(trigrams_freq)
    trigrams_pd.to_csv('trigrams.csv')

    # log t_scores
    trigrams_t_score_pd = trigrams_to_pandas(t_score_trigrams)
    trigrams_t_score_pd.to_csv('trigrams_t_score.csv')

    # use nltk to get collocations
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    text = data_to_text(data, 'lemma')
    tokens = nltk.word_tokenize(text,'english',True)
    text = nltk.Text(tokens)

    finder_thr = TrigramCollocationFinder.from_words(text)

    # compare t_score
    print(finder_thr.nbest(trigram_measures.student_t, 10))
    print(trigrams_t_score_pd.iloc[:10])


if __name__ == '__main__':
    main()
    
