from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from nltk.corpus import stopwords
import regex
from tqdm.notebook import tqdm
from collections import Counter, OrderedDict

# %% [markdown]
# ## Parse texts into df

# %%
# filter_out_tokens = ['\n', '.', ',', '!', '?', '...', ':', ';']
topics = ['Action', 'Drama', 'Crime', 'Biography', 'Comedy']
stop_words = set(stopwords.words('english'))

def read_data(subset="train", squeeze_text=True, collect_analysis=True):

    # if squeeze_text=True, then we won't split text by sentences and each text will consist with one array of tokens
    # if squeeze_text=False, then wil be created corpus of all sentences
    
    token_frequency = Counter()
    term_document_matrix = Counter()
    term_document_matrix_v2 = dict()
    
    texts = dict()
    
    # Перебираем папки
    for folder in topics:
        # Путь к папке
        folder_path = os.path.join(f'../assets/annotated_corpus/', subset, folder)
        # Перебираем файлы в папке
        for file in os.listdir(folder_path):
            # Если это tsv файл
            if file.endswith('.tsv'):
                # Путь к файлу
                file_path = os.path.join(folder_path, file)
                # Читаем файл
                try: 
                    df = pd.read_csv(file_path, sep='\t', header=None)
                except pd.errors.EmptyDataError:
                    continue
                    
                # Группируем токены по предложениям (предполагает, что предложение отделено пустой строкой)
                text = list()
                sentence = list()
                tokens_list = df[0].tolist()
                for token in tokens_list:
                    token = str(token).lower()
                    
                    if squeeze_text:
                        if token == 'nan':
                            continue
                        text.append(token)
                    else:
                        if token == 'nan':
                            if len(sentence) > 0:
                                text.append(sentence)
                            sentence = []
                            continue
                        else:
                            sentence.append(token)
                    
                    if collect_analysis:
                        token_frequency[token] += 1
                        doc_name = f"{folder}_{file.rsplit('.', 1)[0]}"
                        term_document_matrix[(token, doc_name)] += 1
                        if token not in term_document_matrix_v2:
                            term_document_matrix_v2[token] = { doc_name: 1 }
                        else:
                            if doc_name not in term_document_matrix_v2[token]:
                                term_document_matrix_v2[token][doc_name] = 1
                            else:
                                term_document_matrix_v2[token][doc_name] += 1
                
                if not squeeze_text and len(sentence) > 0:
                    text.append(sentence)
                
                if len(text) > 0:
                    texts[f"{folder}_{file.rsplit('.', 1)[0]}"] = text
                    
    return texts, token_frequency, term_document_matrix, term_document_matrix_v2

# %%
train_texts, token_frequency, term_document_matrix, term_document_matrix_v2 = read_data("train", True, True)
test_texts, _, _, _ = read_data("test", True, False)
train_texts

# %%
pure_texts = list(train_texts.values())
pure_texts

# %%
model = Word2Vec(sentences=pure_texts, vector_size=128, window=4, min_count=1, workers=4)
model.train(pure_texts, total_examples=len(pure_texts), epochs=50)

# %%


# %%
from scipy import spatial

def cosine_sim_lib(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


# %%
words = list(model.wv.key_to_index)
X = [model.wv[word] for word in words]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=500)


words = ['cut', 'zoom', 'pan', 'bowl', 'say', 'scream', 'whisper', 'knife', 'sword', 'fork', 'punch', 'hit', 'look', 'bleed', 'fade']

vectors = [model.wv[word] for word in words]

for vec, name in zip(vectors,
                     words):
    reduced_vec = pca.transform(vec[np.newaxis, ...])
    plt.plot(reduced_vec[0][0], reduced_vec[0][1], 'o', color='red')
    plt.annotate(name, (reduced_vec[0][0], reduced_vec[0][1]))
    
"""
for vec, name in zip([vector1, vector2, vector3, vector4], ['cut', 'pan', 'zoom', 'fade']):
    reduced_vec = pca.transform(vec[np.newaxis, ...])
    plt.plot(reduced_vec[0][0], reduced_vec[0][1], 'o', color='blue')
    plt.annotate(name, (reduced_vec[0][0], reduced_vec[0][1]))
    
for vec, name in zip([vector1, vector6, vector4], ['cut', 'hit', 'fade']):
    reduced_vec = pca.transform(vec[np.newaxis, ...])
    plt.plot(reduced_vec[0][0], reduced_vec[0][1], 'o', color='green')
    plt.annotate(name, (reduced_vec[0][0], reduced_vec[0][1]))
"""

# %%
cosine_sim_lib(model.wv["zoom"], model.wv["pan"])

# %%
cosine_sim_lib(model.wv["blade"], model.wv["sword"])

# %%
cosine_sim_lib(model.wv["fork"], model.wv["pan"])

# %%
cosine_sim_lib(model.wv["fork"], model.wv["spoon"])

# %%
cosine_sim_lib(model.wv["cry"], model.wv["scream"])

# %%
cosine_sim_lib(model.wv["say"], model.wv["scream"])

# %%
cosine_sim_lib(model.wv["say"], model.wv["whisper"])

# %%
model.save("../assets/models/w2v")

# %%
def clear_texts(texts, rare_tokens):
    texts_copy = OrderedDict()
    for key, text in texts.items():
        text_copy = list()
        for sentence in text:
            sentence_copy = list()
            for word in sentence:
                if word not in rare_tokens:
                    sentence_copy.append(word)
            if len(sentence_copy) > 0:
                text_copy.append(sentence_copy)
        if len(text_copy) > 0:
            texts_copy[key] = text_copy
    return texts_copy

# %%
from sklearn.preprocessing import normalize

def vectorize_docs(subset, clear_rare_tokens):
    model = Word2Vec.load('../assets/models/w2v')
    texts, _, _, _ = read_data(subset, False, False)
    texts = OrderedDict(texts)

    if clear_rare_tokens:
        print("Delete rare tokens in token_frequency")
        rare_tokens = dict(filter(lambda x: x[1] < 3, token_frequency.items())).keys()

        print("Clear texts")
        texts = clear_texts(texts, rare_tokens)

    print("Vectorize texts")
    model_data = np.full((len(texts), 128), 0, dtype=np.float32)
    for i, (doc, sentences) in enumerate(texts.items()):
        temp = np.empty((0, 128), dtype=np.float32)
        cnt = 0
        for sentence in sentences:
            sentence_vec = np.full(128, 0, dtype=np.float32)
            for word in sentence:
                try:
                    sentence_vec += model.wv[word]
                except KeyError: 
                    continue
            temp = np.vstack((temp, sentence_vec))

        model_data[i] = normalize(np.mean(temp, axis=0).reshape(1, -1))

    return texts, model_data

# %%
texts, test_vecs = vectorize_docs("test", clear_rare_tokens=True)
test_vecs.shape

# %%
train_texts_v2, train_vecs = vectorize_docs("train", clear_rare_tokens=True)
train_vecs.shape

# %%
import csv
def write_data(texts, vectors, subset="test"):
    with open(f'../assets/annotated_corpus/{subset}-embeddings.tsv', 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for i, doc_name in enumerate(texts.keys()):
            writer.writerow([doc_name] + vectors[i].tolist())

# %%
write_data(texts, test_vecs, subset="test")

# %%
write_data(train_texts_v2, train_vecs, subset="train")


