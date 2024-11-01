import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from collections import defaultdict
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
nltk.download('stopwords')

sentences = list()
sentence = list()
with open('/content/annotations (5).tsv') as file:
  for line in file:
      if line != "\n":
        sentence.append(line.split('\t')[1])
      if line == "\n" and sentence:
        sentences.append(sentence)
        sentence = list()

stop_words = set(stopwords.words('english'))
clear_data = []

def clean_sentence(sentence):
    cleaned = []
    for lemma in sentence:
        cleaned_tek = re.sub(r"[^\w\s]|[\d]", "", lemma.lower())
        if len(cleaned_tek) != len(lemma) or len(cleaned_tek) <3:
          continue;

        if cleaned_tek and cleaned_tek not in stop_words:
            cleaned.append(cleaned_tek)
    return cleaned

clear_data = []  # Инициализация списка для хранения очищенных данных

for sentence in sentences:
    clear_data.append(clean_sentence(sentence))

from gensim.models import Word2Vec
w2v = Word2Vec(sentences=clear_data, epochs=70, window=4, min_count=3)

# Получение всех слов
words = list(w2v.wv.key_to_index.keys())
print("Все слова в модели:", words)

word_vector = w2v.wv['win'] 
print("Вектор для слова 'win':", word_vector)



def cosine_distance(a, b):
    return (1 - np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))) / 2

word = "computer"
similar_words = ["software", "machine", "hardware"]  # Семантически близкие
related_words = ["program", "cpu", "system"]   # Слова из той же предметной области
distant_words = ["god", "religion", "church"]  # Семантически далекие

for similar_word in similar_words:
    print(f"Косинусное расстояние между {word} и {similar_word}: {cosine_distance(w2v.wv[word], w2v.wv[similar_word])}")

for related_word in related_words:
    print(f"Косинусное расстояние между {word} и {related_word}: {cosine_distance(w2v.wv[word], w2v.wv[related_word])}")

for distant_word in distant_words:
    print(f"Косинусное расстояние между {word} и {distant_word}: {cosine_distance(w2v.wv[word], w2v.wv[distant_word])}")


def plot_word_vectors(words, vector_data):
    #получение двумерного представления
    pca_model = PCA(n_components=2)
    reduced_vectors = pd.DataFrame(pca_model.fit_transform([vector_data[word] for word in words]))

    reduced_vectors.index = words
    reduced_vectors.columns = ["x_coord", "y_coord"]

    scatter_plot = sns.scatterplot(data=reduced_vectors, x="x_coord", y="y_coord")

    # Добавление меток для каждой точки
    for word in reduced_vectors.index:
        coordinates = reduced_vectors.loc[word]
        scatter_plot.text(coordinates.x_coord, coordinates.y_coord, word)

    return scatter_plot

words_to_plot = ["car", "bike", "jesus", "road", "religion", "computer"]
plot_word_vectors(words_to_plot, w2v.wv)


import numpy as np
def calculate_vectors(sentences, w2v):
    final_vector = np.zeros(w2v.vector_size)

    for sentence in sentences:
        current_sentence_vector = np.zeros(w2v.vector_size)

        for word in sentence:
            if word in w2v.wv.key_to_index:
                current_sentence_vector += w2v.wv[word]

        if len(sentence) > 0:
            current_sentence_vector /= len(sentence)

        final_vector += current_sentence_vector

    if len(sentences) > 0:
        final_vector /= len(sentences)

    return final_vector

calculate_vectors(clear_data, w2v)

import os
import chardet
from scipy.sparse import csr_matrix, save_npz
def detect_encoding(file_path):

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']
def process_file(file_path):

    sentences = []
    sentence = []

    with open(file_path, 'r', encoding=detect_encoding(file_path)) as file:
        for line in file:
            if line != "\n":
                sentence.append(line.split('\t')[1])
            if line == "\n" and sentence:
                sentences.append(sentence)
                sentence = []

    clear_data = []
    for sent in sentences:
        clear_data.append(clean_sentence(sent))

    return clear_data

def save_vectors_to_tsv(file_vectors, output_tsv_path):

    with open(output_tsv_path, 'w', encoding='utf-8') as f:
        for doc_id, vector in file_vectors.items():
            vector_str = '\t'.join(map(str, vector))
            f.write(f"{doc_id}\t{vector_str}\n")


def process_directory(directory_path, output_tsv_path):

    file_vectors = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory_path, filename)
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            clear_data = process_file(file_path)
            document_vector = calculate_vectors(clear_data, w2v)
            file_vectors[doc_id] = document_vector

    save_vectors_to_tsv(file_vectors, output_tsv_path)
    return file_vectors


directory = '/content/space'


doc_vec = process_directory(directory, 'space_vec.tsv')

flat_list = [item for sublist in clear_data for item in sublist]
unique_words = list(set(flat_list))

from collections import defaultdict
import os
from scipy.sparse import csr_matrix

def build_token_dictionary_frec(tokens):
  for token in tokens:
      token_freqs[token] += 1
  return token_freqs


token_freqs = build_token_dictionary_frec(flat_list)


def filter_low_frequency_tokens(token_freqs, min_freq=5):
    return {token: freq for token, freq in token_freqs.items() if freq >= min_freq}

filtered_token_freqs = filter_low_frequency_tokens(token_freqs)


def save_token_frequencies(token_freqs, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, freq in token_freqs.items():
            f.write(f"{token}\t{freq}\n")

save_token_frequencies(filtered_token_freqs, 'token_frequencies.tsv')

import os
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix

def process_directory(directory_path):
    doc_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory_path, filename)
            doc_id = os.path.splitext(filename)[0]  # Удаляем расширение
            doc_texts[doc_id] = [item for sublist in process_file(file_path) for item in sublist]
    return doc_texts

def create_term_doc_matrix(doc_texts):
    # Собираем все уникальные токены
    token_freq = defaultdict(int)
    for doc_id, tokens in doc_texts.items():
        for token in tokens:
            token_freq[token] += 1
    unique_tokens = list(token_freq.keys())

    # Маппинг токенов в индексы
    token_index = {token: i for i, token in enumerate(unique_tokens)}

    # Создаем разреженную матрицу для хранения данных
    rows = []
    cols = []
    data = []

    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_texts.keys())}  # Маппинг документов в индексы

    for doc_id, tokens in doc_texts.items():
        doc_idx = doc_index[doc_id]
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in token_index:
                token_idx = token_index[token]
                rows.append(doc_idx)
                cols.append(token_idx)
                data.append(count)

    term_doc_matrix = csr_matrix((data, (rows, cols)), shape=(len(doc_texts), len(unique_tokens)))

    return term_doc_matrix, doc_index, token_index, token_freq

def save_term_doc_matrix_to_tsv(term_doc_matrix, doc_index, token_index, output_tsv_path, unique_tokens):
    dense_matrix = term_doc_matrix.toarray()
    with open(output_tsv_path, 'w', encoding='utf-8') as tsv_file:
        # Записываем заголовок
        header = ['doc_id'] + unique_tokens
        tsv_file.write('\t'.join(header) + '\n')

        # Записываем строки
        for doc_id, doc_idx in doc_index.items():
            row = [doc_id] + list(map(str, dense_matrix[doc_idx]))
            tsv_file.write('\t'.join(row) + '\n')

        # Вычисляем сумму для каждого токена и записываем в конец
        token_sums = dense_matrix.sum(axis=0)  # A1 преобразует разреженный массив в обычный одномерный
        sum_row = ['Total'] + list(map(str, token_sums))
        tsv_file.write('\t'.join(sum_row) + '\n')

directory_path = '/content/electronics'
output_tsv_path = 'term_doc_matrix.tsv'

doc_texts = process_directory(directory_path)

term_doc_matrix, doc_index, token_index, token_freq = create_term_doc_matrix(doc_texts)
unique_tokens = list(token_freq.keys())
save_term_doc_matrix_to_tsv(term_doc_matrix, doc_index, token_index, output_tsv_path, unique_tokens)