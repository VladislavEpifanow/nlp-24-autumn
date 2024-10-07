import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from projects.yelp_labeller.source.association_meter.main import apply_steps, steps
from projects.yelp_labeller.source.classifier.main import load_dataset
from projects.yelp_labeller.source.classifier.tokenizer import tokenize
from projects.yelp_labeller.source.vectorizer.task_3 import get_token_vector


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


def process_df(df: pd.DataFrame, save_path: str, w2v_model: Word2Vec):
    file = open(save_path, "w")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        label, text = row["label"], row["text"]
        doc_name = f"{idx:06d}"
        text_emb = get_text_embedding(text, w2v_model)

        emb_to_text = "\t".join(map(str, text_emb.tolist()))
        file.write(f"{doc_name}\t{emb_to_text}\n")
    file.close()


if __name__ == "__main__":
    model = Word2Vec.load("w2v_model")

    split_type: str = "test"
    n: int | None = None
    save_path = f"{split_type}_embeddings.tsv"
    dataset_path = r"..\..\assets\{split_type}.csv"

    df = load_dataset(split_type, n=n, dataset_path=dataset_path)
    process_df(df, save_path, w2v_model=model)
