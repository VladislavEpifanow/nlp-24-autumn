import json
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

np.random.seed(42)


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
        # self.w2 = np.random.uniform(-0.8, 0.8, (self.embedding_size, self.vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w1.T)
        y = self.softmax(u)
        return y, h, u

    def onehot(self, word_idx: int):
        x = np.zeros(self.vocab_size)
        x[word_idx] = 1
        return x

    def generate_training_data(self, sentences):
        x = []
        y = []

        for sentence in tqdm(sentences, desc="Generating train data"):
            sentence_idxs = [self.word_index[word] for word in sentence]
            for pos, word_idx in enumerate(sentence_idxs):
                x.append(self.onehot(word_idx))

                context = []
                # Добавляем контекст
                start = max(0, pos - self.window_size)
                end = min(len(sentence_idxs), pos + self.window_size + 1)
                for context_pos in range(start, end):
                    if context_pos != pos:
                        context.append(self.onehot(sentence_idxs[context_pos]))
                y.append(np.sum(np.asarray(context), axis=0))
        return np.asarray(x), np.asarray(y)

    # def backward(self, e, h, x):
    #     dw2 = np.outer(h, e)
    #     dw1 = np.outer(x, np.dot(self.w1, e.T))
    #     self.w1 = self.w1 - (self.learning_rate * dw1)
    #     self.w1 = self.w1 - (self.learning_rate * dw2)

    def backward(self, e, u, h, x):
        dw2 = np.outer(h, e).T
        dw1 = np.outer(x, np.dot(self.w1.T, e))
        self.w1 = self.w1 - (self.learning_rate * dw1)
        self.w1 = self.w1 - (self.learning_rate * dw2)

    def train(self, x, y):
        self.initialize_weights()
        for epoch in tqdm(range(self.epochs), desc="training"):
            for word_oh, context in zip(x, y):
                y_pred, h, u = self.forward(word_oh)
                e = context - y_pred
                self.backward(e, u, h, word_oh)

    def dump(self, file_name):
        np.save(file_name, self.w1, allow_pickle=False)

    def load(self, file_name):
        self.w1 = np.load(file_name)

    def get_embedding(self, word: str):
        word_idx = self.word_index.get(word, None)
        if not word_idx:
            return
        return self.w1[word_idx, :]


if __name__ == "__main__":
    cache_name = r"./projects/yelp_labeller/source/association_meter/cache"
    data = json.load(open(cache_name, "r"))
    # data = [["I", "love", "NLP", "and", "word2vec"], ["I", "love", "deep", "learning"], ["NLP", "is", "fun"]]

    embedding_size = 100
    window_size = 5
    lr = 0.025
    epochs = 5

    word2vec = Word2Vec(embedding_size=embedding_size, window_size=window_size, learning_rate=lr, epochs=epochs)
    logger.info("Building vocab")
    word2vec.build_vocab(data)
    logger.info("generating_data")
    x, y = word2vec.generate_training_data(data)
    np.save("x_data.npy", x, allow_pickle=False)
    np.save("y_data.npy", y, allow_pickle=False)
    word2vec.train(x, y)
    file_name = f"data_dump_e={epochs}_lr={lr}.npy"
    word2vec.dump(file_name)
    # word2vec.load(file_name)
