import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from numpy.linalg import norm
from tqdm import tqdm

from projects.yelp_labeller.source.association_meter.main import apply_steps, steps
from projects.yelp_labeller.source.classifier.tokenizer import tokenize
from projects.yelp_labeller.source.vectorizer.task_3_4 import get_token_vector


def get_text_embedding(text: str, w2v_model: Word2Vec) -> np.array:
    tokens = tokenize(text)
    text_emb = None
    for sent in tokens:
        cleared_sent = list(filter(lambda x: x, [apply_steps(word, steps) for word in sent]))
        if not cleared_sent:
            continue
        sent_emb = []
        for word in cleared_sent:
            sent_emb.append(get_token_vector(word, w2v_model))
        if text_emb is None:
            text_emb = np.append(np.array([[]]), [np.mean(np.asarray(sent_emb), axis=0)], axis=1)
        else:
            text_emb = np.append(text_emb, [np.mean(np.asarray(sent_emb), axis=0)], axis=0)

    return np.mean(text_emb, axis=0) if text_emb is not None else np.zeros(w2v_model.vector_size)

def _get_text_emb(sents: list[str], w2v_model: Word2Vec) -> np.array:
    if not sents:
        return np.zeros(w2v_model.vector_size)
    words_emb = [get_token_vector(word, w2v_model) for word in sents]
    sent_emb = np.mean(np.asarray(words_emb), axis=0)
    return sent_emb if words_emb and sent_emb is not None else np.zeros(w2v_model.vector_size)

def process_df(df: pd.DataFrame, save_path: str, w2v_model: Word2Vec):
    file = open(save_path, "w")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        label, text = row["label"], row["text"]
        doc_name = f"{idx:06d}"
        text_emb = get_text_embedding(text, w2v_model)

        emb_to_text = "\t".join(map(str, text_emb.tolist()))
        file.write(f"{doc_name}\t{emb_to_text}\n")
    file.close()

def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_words_cosine(w2v_model: Word2Vec, words: list[str]):
    main_word = words[0]
    main_emb = get_token_vector(main_word, w2v_model)
    cosine_dict = {main_word: {}}
    for word in words[1:]:
        word_emb = get_token_vector(word, w2v_model)
        cosine_val = cosine(main_emb, word_emb)
        cosine_dict[main_word][word] = cosine_val
    return cosine_dict


def get_cosine_distance_groups(word2vec):
    words_g1 = ["put", "give", "grass", "slow"]
    words_g2 = ["good", "best", "nice", "bad", "worst", "terrible", "awful"]
    words_g3 = ["beer", "water", "drink", "mojito", "window", "license", "office"]

    cosine_dict = {}
    for group in [words_g1, words_g2, words_g3]:
        group_cosine = get_words_cosine(word2vec, words=group)
        cosine_dict.update(group_cosine)

    with open("cosine_distance.txt", "w") as file:
        for word, dists in cosine_dict.items():
            file.write(word + "\n")
            file.write("\n".join(["\t".join([key, str(value)]) for key, value in dists.items()]))
            file.write("\n\n")


def process_data(data: list[list[str]], save_path: str, w2v_model: Word2Vec):
    file = open(save_path, "w")
    for idx, text in tqdm(enumerate(data), total=len(data)):
        doc_name = f"{idx:06d}"
        text_emb = _get_text_emb(text, w2v_model)

        emb_to_text = "\t".join(map(str, text_emb.tolist()))
        file.write(f"{doc_name}\t{emb_to_text}\n")
    file.close()


if __name__ == "__main__":
    n: int | None = 100_000
    model_path = "w2v_model_full"
    save_path = "{split_type}_embeddings.tsv"
    dataset_path = "../../assets/{split_type}.csv"
    cache_name = "../../assets/cache_{split_type}"

    seed = 1
    window = 5
    sg = 0  # use CBOW
    cbow_mean = 0
    min_count = 1
    vector_size = 100

    with open(cache_name.format(split_type="train"), "r") as file:
        data = json.load(file)

    # Обучение модели
    # model = Word2Vec(data, seed=seed, window=window, sg=sg, cbow_mean=cbow_mean, min_count=min_count,
    #                  vector_size=vector_size)
    # model.save(model_path)

    # Импорт модели
    model = Word2Vec.load(model_path)

    # Обработка TRAIN
    # process_data(data, save_path.format(split_type="train"), w2v_model=model)

    # Обработка TEST
    # df = load_dataset("test", dataset_path=dataset_path.format(split_type="test"))
    # with open(cache_name.format(split_type="test"), "r") as file:
    #     data = json.load(file)
    # process_data(data, save_path.format(split_type="test"), w2v_model=model)

    get_cosine_distance_groups(model)