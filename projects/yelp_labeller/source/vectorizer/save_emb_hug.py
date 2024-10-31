import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from projects.yelp_labeller.source.classifier.main import load_dataset
from projects.yelp_labeller.source.classifier.tokenizer import split_to_sentence


def get_text_embedding(text: str, w2v_model) -> np.array:
    sent_emb = []
    for sent in split_to_sentence(text):
        sent_emb.append(get_token_vector(sent, w2v_model))

    return np.mean(np.asarray(sent_emb), axis=0) if sent_emb is not None else np.zeros(w2v_model.vector_size)


def process_df(df: pd.DataFrame, save_path: str, w2v_model):
    file = open(save_path, "w")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        label, text = row["label"], row["text"]
        doc_name = f"{idx:06d}"
        text_emb = get_text_embedding(text, w2v_model)

        emb_to_text = "\t".join(map(str, text_emb.tolist()))
        file.write(f"{doc_name}\t{emb_to_text}\n")
    file.close()


def get_token_vector(text: str, model):
    return model.encode(text)


if __name__ == "__main__":
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    split_type: str = "test"
    n: int | None = 5_000
    save_path = f"{split_type}_embeddings_hug_{n}.tsv"
    dataset_path = r"..\..\assets\{split_type}.csv"

    df = load_dataset(split_type, n=n, dataset_path=dataset_path)
    process_df(df, save_path, w2v_model=model)
