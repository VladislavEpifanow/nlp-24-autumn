import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from projects.yelp_labeller.source.association_meter.main import data_iterator, apply_steps, steps
from projects.yelp_labeller.source.classifier.main import load_dataset
from projects.yelp_labeller.source.classifier.tokenizer import tokenize
from projects.yelp_labeller.source.vectorizer.main import Word2Vec


def get_text_embedding(text: str, w2v_model: Word2Vec) -> np.array:
    tokens = tokenize(text)
    text_emb = None
    for sent in tokens:
        cleared_sent = list(filter(lambda x: x, [apply_steps(word, steps) for word in sent]))
        sent_emb = w2v_model.get_embeddings(cleared_sent)
        if text_emb is None:
            text_emb = np.append(np.array([[]]), [np.mean(sent_emb, axis=0)], axis=1)
        else:
            text_emb = np.append(text_emb, [np.mean(sent_emb, axis=0)], axis=0)

    return np.mean(text_emb, axis=0)


def process_df(df: pd.DataFrame, save_path: str, w2v_model: Word2Vec):
    embeddings = []
    file = open(save_path, "w")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        label, text = row["label"], row["text"]
        doc_name = f"{idx:06d}"

        text_emb = get_text_embedding(text, w2v_model)
        emb_shape = text_emb.shape

        emb_to_text = "\t".join(map(str, text_emb.tolist()))
        file.write(f"{doc_name}\t{emb_to_text}\n")
    file.close()

if __name__ == "__main__":
    embedding_size = 100
    window_size = 5
    lr = 0.00025
    epochs = 5
    limit = 5

    word2vec = Word2Vec(embedding_size=embedding_size, window_size=window_size, learning_rate=lr, epochs=epochs)

    cache_name = r"./projects/yelp_labeller/source/association_meter/cache"
    with open(cache_name, "r") as file:
        data = json.load(file)
    word2vec.build_vocab(data)
    file_name = f"data_dump_limit={limit}_e={epochs}_lr={lr}.npy"
    word2vec.load(file_name)

    text = "I like health insurance. love dr"

    # print(get_text_embedding(text, word2vec))

    split_type: str = "test"
    n: int | None = 5
    save_path = f"{split_type}_embeddings.tsv"
    dataset_path = r".\projects\yelp_labeller\assets\{split_type}.csv"

    df = load_dataset(split_type, n=n, dataset_path=dataset_path)
    process_df(df, save_path, w2v_model=word2vec)
