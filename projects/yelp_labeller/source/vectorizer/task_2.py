import json
from collections import defaultdict

import numpy as np

from projects.yelp_labeller.source.association_meter.main import apply_steps, steps
from projects.yelp_labeller.source.classifier.tokenizer import tokenize


def text_to_tfidf(text: str, word_index: dict[str, int], term_document_matrix):
    tokens = tokenize(text)

    tfidf_vector = np.zeros(len(word_index))

    token_count = defaultdict(int)
    for sent in tokens:
        cleared_sent = list(filter(lambda x: x, [apply_steps(word, steps) for word in sent]))
        if not cleared_sent:
            continue
        for token in cleared_sent:
            if token in word_index:
                token_count[token] += 1

    token_sum = sum(token_count.values())

    for token, frequency in token_count.items():
        token_index = word_index[token]
        if token_index >= len(term_document_matrix):
            continue
        tfidf_vector[token_index] = frequency / token_sum * np.log(
            len(term_document_matrix[0]) / np.count_nonzero(term_document_matrix[:, token_index]))

    return tfidf_vector


text = "I love eating pizza and pasta in this place!"

with open("task_1_word_index.json", "r") as file:
    word_index = json.load(file)
td_matrix = np.loadtxt("task_1_td.ch")

tfidf_vector = text_to_tfidf(text, word_index, td_matrix)

print(tfidf_vector)

np.savetxt('task_2_tfidf_vector.txt', tfidf_vector, delimiter=',', fmt='%d')
