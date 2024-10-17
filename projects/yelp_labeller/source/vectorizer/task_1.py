import json
from collections import defaultdict

import numpy as np


def get_td_array(doc_dict, total_words_cnt: int):
    doc_arr = [0 for _ in range(total_words_cnt)]
    for word_idx, word_cnt, in doc_dict.items():
        doc_arr[word_idx] = word_cnt
    return doc_arr


if __name__ == "__main__":
    cache_name = r"../../assets/cache_train"
    n = 5_000

    with open(cache_name, "r") as file:
        data = json.load(file)
    if n:
        data = data[:n]

    word_counts = defaultdict(int)

    for idx, sentence in enumerate(data):
        for word in sentence:
            word_counts[word] += 1

    with open("task_1_word_counts.json", "w") as file:
        json.dump(word_counts, file)

    word_index = {word: i for i, word in enumerate(word_counts.keys())}

    with open("task_1_word_index.json", "w") as file:
        json.dump(word_index, file)

    td_matrix = np.zeros((len(data), len(word_index)))

    for idx, sentence in enumerate(data):
        for word in sentence:
            td_matrix[idx, word_index[word]] += 1

    np.savetxt("task_1_td.ch", td_matrix, fmt='%d')
