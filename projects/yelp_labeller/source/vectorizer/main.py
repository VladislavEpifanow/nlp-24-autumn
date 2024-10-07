import json
import logging
import os.path
from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

np.random.seed(42)

import warnings

# warnings.filterwarnings("error")


# Skip-gram
class Word2Vec:
    def __init__(self, embedding_size=100, window_size=2, learning_rate=0.025, epochs=5):
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.word_counts = defaultdict(int)  # word: cnt
        self.word_index = {}  # word: idx
        self.index_word = {}  # idx:  word
        self.vocab_size = 0
        self.w1: None | np.ndarray = None

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
        self.word_index = {word: i for i, word in enumerate(self.word_counts.keys())}
        self.index_word = {i: word for word, i in self.word_index.items()}
        self.vocab_size = len(self.word_index)

    def initialize_weights(self):
        self.w1 = np.random.uniform(-0.8, 0.8, (self.vocab_size, self.embedding_size))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.embedding_size, self.vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        sum_val = e_x.sum(axis=0)
        return e_x / sum_val if sum_val else 0.0

    # def forward(self, x):
    #     h = np.dot(x, self.w1)
    #     u = np.dot(h, self.w1.T)
    #     y = self.softmax(u)
    #     return y, h, u

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        # y = self.softmax(u)
        y = self.sigmoid(u)
        return y, h, u

    # def backward(self, e, h, x):
    #     dw2 = np.outer(h, e)
    #     dw1 = np.outer(x, np.dot(self.w1, e.T))
    #     self.w1 = self.w1 - (self.learning_rate * dw1)
    #     self.w1 = self.w1 - (self.learning_rate * dw2)

    def backward(self, e, h, x):
        dw2 = np.outer(h, e)
        dw1 = np.outer(x, self.w2 @ e)
        self.w2 = self.w2 - (self.learning_rate * dw2)
        self.w1 = self.w1 - (self.learning_rate * dw1)

    def train(self, x, y):
        self.initialize_weights()
        for epoch in tqdm(range(self.epochs), desc="training"):
            for sent_idx, sent_data in enumerate(zip(x, y)):
                word_oh, context = sent_data
                x_vector = self.onehot(word_oh)
                y_vector = self.onehot(context)
                y_pred, h, u = self.forward(x_vector)
                e = y_vector - y_pred
                self.backward(e, h, x_vector)

    def onehot(self, word_idx: int | list[int]):
        x = np.zeros(self.vocab_size)
        if isinstance(word_idx, int):
            x[word_idx] = 1
        elif isinstance(word_idx, list):
            for word_idx_ in word_idx:
                x[word_idx_] = 1
        return x

    def generate_training_data(self, sentences, limit: int = 0):
        x = []
        y = []
        if limit > 0:
            sentences = sentences[:limit]
        for sentence in tqdm(sentences, desc="Generating train data"):
            sentence_idxs = [self.word_index[word] for word in sentence]
            for pos, word_idx in enumerate(sentence_idxs):
                x.append(word_idx)

                context = []
                # Добавляем контекст
                start = max(0, pos - self.window_size)
                end = min(len(sentence_idxs), pos + self.window_size + 1)
                for context_pos in range(start, end):
                    if context_pos != pos:
                        context.append(sentence_idxs[context_pos])
                if not context:
                    continue
                y.append(context)
        return x, y

    def dump(self, file_name: str):
        np.save("w1_" + file_name, self.w1, allow_pickle=False)
        np.save("w2_" + file_name, self.w1, allow_pickle=False)

    def load(self, file_name: str):
        self.w1 = np.load("w1_" + file_name)
        self.w2 = np.load("w2_" + file_name)

    def get_embedding(self, word: str):
        word_idx = self.word_index.get(word, None)
        if word_idx is None:
            return
        return self.w1[word_idx, :]

    def get_embeddings(self, words: list[str]):
        word_idxs = []
        for word in words:
            word_idx = self.word_index.get(word, None)
            if word_idx is None:
                continue
            word_idxs.append(word_idx)
        return self.w1[word_idxs, :]


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_words_cosine(w2v_model: Word2Vec, words: list[str]):
    main_word = words[0]
    main_emb = w2v_model.get_embedding(main_word)
    cosine_dict = {main_word: {}}
    for word in words[1:]:
        word_emb = w2v_model.get_embedding(word)
        cosine_val = cosine(main_emb, word_emb)
        cosine_dict[main_word][word] = cosine_val
    return cosine_dict


def get_cosine_distance_groups(word2vec):
    words_g1 = ["like", "love", "hate"]
    words_g2 = ["good", "best", "nice", "bad", "worst", "terrible", "awful"]
    words_g3 = ["beer", "water", "drink", "mojito", "window", "dr", "weather"]

    cosine_dict = {}
    for group in [words_g1, words_g2, words_g3]:
        group_cosine = get_words_cosine(word2vec, words=group)
        cosine_dict.update(group_cosine)

    with open("cosine_distance.txt", "w") as file:
        for word, dists in cosine_dict.items():
            file.write(word + "\n")
            file.write("\n".join(["\t".join([key, str(value)]) for key, value in dists.items()]))
            file.write("\n\n")


if __name__ == "__main__":
    cache_name = r"./projects/yelp_labeller/source/association_meter/cache"
    with open(cache_name, "r") as file:
        data = json.load(file)
    # data = [["I", "love", "NLP", "and", "word2vec"], ["I", "love", "deep", "learning"], ["NLP", "is", "fun"]]

    embedding_size = 100
    window_size = 5
    lr = 1e-5
    epochs = 5
    limit = 0  # 0 = no limit
    weights_dump_file_name = f"data_dump_limit={limit}_e={epochs}_lr={lr}.npy"

    word2vec = Word2Vec(embedding_size=embedding_size, window_size=window_size, learning_rate=lr, epochs=epochs)
    logger.info("Building vocab")
    word2vec.build_vocab(data)

    if os.path.exists(weights_dump_file_name):
        logger.info("Loading weights")
        word2vec.load(weights_dump_file_name)
    else:
        x_data_file_name = f"x_data_limit={limit}.ch"
        y_data_file_name = f"y_data_limit={limit}.ch"
        if not os.path.exists(x_data_file_name) or not os.path.exists(y_data_file_name):
            logger.info("Generating data")
            x, y = word2vec.generate_training_data(data, limit=limit)
            logger.info("saving_data")
            with open(x_data_file_name, "w") as file:
                json.dump(x, file)
            with open(y_data_file_name, "w") as file:
                json.dump(y, file)
        else:
            logger.info("Loading data")
            with open(x_data_file_name, "r") as file:
                x = json.load(file)
            with open(y_data_file_name, "r") as file:
                y = json.load(file)
        logger.info("Training model")
        word2vec.train(x, y)
        logger.info("Saving weights")
        word2vec.dump(weights_dump_file_name)
