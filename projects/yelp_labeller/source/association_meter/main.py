import itertools
import json
import math
import os
from functools import reduce

import nltk
import regex
from nltk.corpus import stopwords
from tqdm import tqdm

# nltk.download('stopwords')
stopwords = stopwords.words("english")

_product = lambda s: reduce(lambda x, y: x * y, s)


# Шаг 1
def data_iterator(path: str, split_type: str):
    my_path = path.format(split_type=split_type)
    dir_labels = os.listdir(my_path)
    for label in tqdm(dir_labels, desc="dirs", leave=True):
        label_path = os.path.join(my_path, label)
        dir_files = os.listdir(label_path)
        for file_name in tqdm(dir_files, desc="files", leave=True, position=True):
            file_path = os.path.join(label_path, file_name)
            with open(file_path, "r") as file:
                data = [line.strip().split("\t") for line in file.readlines()]
            yield data
    return

steps = []
# Шаг 2
punkt_regex = r"[^\P{P}-]+"
punkt_step = lambda x: regex.sub(punkt_regex, "", x)
steps.append(punkt_step)
# Шаг 3
steps.append(lambda x: x.lower())
# Шаг 4
steps.append(lambda x: x if x not in stopwords else None)

# Шаг 5
word_types = {"token": 0,
              "stemma": 1,
              "lemma": 2}


def apply_steps(word: str, steps: list):
    for step in steps:
        word = step(word)
        if not word:
            return
    return word


def get_n_grams(data: list[str], n: int) -> list[tuple[str, ...]]:
    processed_data = []
    for i in range(0, len(data) - n + 1):
        processed_data.append(tuple(data[i:i + n]))
    return processed_data


def process_data(path: str, split_type: str, word_type: str, limit: int | None = None) -> list[
    list[str]]:
    assert word_type in word_types.keys(), f"word type {word_type} not in word_types {word_types}"
    word_idx = word_types[word_type]
    processed_data = []
    processed_n = 0
    for file_data in data_iterator(path, split_type):
        if limit and processed_n >= limit: break
        processed_file_data = []
        for word_data in file_data:
            word = word_data[word_idx]
            word = apply_steps(word, steps)
            if not word:
                continue
            processed_file_data.append(word)
        processed_data.append(processed_file_data)
        processed_n += 1
    return processed_data


def calc_t_score(data: list[tuple[str, ...]]):
    def product(n_gram):
        value = 1
        for w in n_gram:
            value *= words_stats[w]
        return value

    n_gram_stats = {}
    words_stats = {}
    for n_gram in tqdm(data, desc="calculating t-score"):
        n_gram_stats[n_gram] = n_gram_stats.get(n_gram, 0) + 1
        for w in n_gram:
            words_stats[w] = words_stats.get(w, 0) + 1
    total_num = sum(words_stats.values())
    gram_len = len(n_gram)

    n_gram_score = {}
    for n_gram, value in n_gram_stats.items():
        score = (value - (product(n_gram) / (total_num ** (gram_len - 1)))) / (
                value ** 0.5)  # (value ** (1 / gram_len))
        n_gram_score[n_gram] = score
    return n_gram_score


# def calc_log_likelihood(data: list[tuple[str, ...]]):
#     n_gram_stats = {}
#     words_stats = {}
#     for n_gram in tqdm(data, desc="calculating log-likelihood"):
#         n_gram_stats[n_gram] = n_gram_stats.get(n_gram, 0) + 1
#         for w in n_gram:
#             words_stats[w] = words_stats.get(w, 0) + 1
#     total_num = sum(n_gram_stats.values())
#     gram_len = len(n_gram)
#
#     n_gram_score = {}
#     for n_gram, value in n_gram_stats.items():
#         x_1 = words_stats[n_gram[0]] / total_num
#         x_2 = words_stats[n_gram[1]] / total_num
#         x_3 = words_stats[n_gram[2]] / total_num
#         x_13 = x_1 * x_3
#         x_23 = x_2 * x_3
#         expected = x_13 * x_23
#         score = gram_len * value + math.log(value / expected)
#         n_gram_score[n_gram] = score
#     return n_gram_score


def calc_log_likelihood(data: list[tuple[str, ...]]):
    def product(n_gram):
        value = 1
        for w in n_gram:
            value *= words_stats[w]
        return value

    n_gram_stats = {}
    words_stats = {}
    for n_gram in tqdm(data, desc="calculating log-likelihood"):
        n_gram_stats[n_gram] = n_gram_stats.get(n_gram, 0) + 1
        for w in n_gram:
            words_stats[w] = words_stats.get(w, 0) + 1
    total_num = len(n_gram_stats.keys())
    gram_len = len(n_gram)

    n_gram_score = {}
    for n_gram, value in n_gram_stats.items():
        expected_count = product(n_gram) / (total_num ** (gram_len - 1))
        score = 2 * value * math.log(value / expected_count)
        n_gram_score[n_gram] = score
    return n_gram_score


def save_results(data, path: str):
    with open(path, "w") as file:
        for item in data:
            n_gram, value = item
            file.write(" ".join(n_gram) + f"\t{value}\n")


def trigram_t_score(tokens):
    from nltk import TrigramCollocationFinder
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder_tri = TrigramCollocationFinder.from_words(tokens)
    return finder_tri.score_ngrams(trigram_measures.student_t)


def trigram_log_likelihood(tokens):
    from nltk import TrigramCollocationFinder
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder_tri = TrigramCollocationFinder.from_words(tokens)
    return finder_tri.score_ngrams(trigram_measures.likelihood_ratio)


if __name__ == "__main__":
    path = r'C:\Users\Karpo\PycharmProjects\nlp-24-autumn\projects\yelp_labeller\assets\annotated-corpus\{split_type}'
    split_type = "test"
    word_type = "token"
    n = 3
    dataset_limit = 0
    # mesure = "t-score"
    mesure = "log"

    cache_name = f"{split_type}_tokens_{dataset_limit}"

    if not os.path.exists(cache_name):
        data = process_data(path, split_type, word_type, limit=dataset_limit)
        json.dump(data, open(cache_name, "w"))
    else:
        data = json.load(open(cache_name, "r"))
    nltk_data = list(itertools.chain(*data))
    n_grams = get_n_grams(nltk_data, n)
    if mesure == 't-score':
        t_data = calc_t_score(n_grams)
        t_data = sorted(t_data.items(), key=lambda x: -x[1])[0:30]
        save_results(t_data,
                     r"C:\Users\Alex\PycharmProjects\nlp-24-autumn\projects\yelp_labeller\assets\example\my_t_score_v3.tsv")
        print(t_data[0:30])
        nltk_t_score = trigram_t_score(nltk_data)
        print(nltk_t_score[0:30])
        save_results(nltk_t_score[0:30],
                     r"C:\Users\Alex\PycharmProjects\nlp-24-autumn\projects\yelp_labeller\assets\example\nltk_t_score_v3.tsv")
    elif mesure == "log":
        t_data = calc_log_likelihood(n_grams)
        t_data = sorted(t_data.items(), key=lambda x: -x[1])[0:30]
        save_results(t_data,
                     r"C:\Users\Alex\PycharmProjects\nlp-24-autumn\projects\yelp_labeller\assets\example\my_log_v3.tsv")
        print(t_data[0:30])
        nltk_data = itertools.chain(*data)
        nltk_t_score = trigram_log_likelihood(nltk_data)
        print(nltk_t_score[0:30])
        save_results(nltk_t_score[0:30],
                     r"C:\Users\Alex\PycharmProjects\nlp-24-autumn\projects\yelp_labeller\assets\example\nltk_log_v3.tsv")
