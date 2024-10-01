import os.path

import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer
from tqdm.auto import tqdm

from projects.yelp_labeller.source.classifier.tokenizer import tokenize


# nltk.download('wordnet')

def load_dataset(split_type="train", n: int | None = None):
    assert split_type == "train" or split_type == "test"
    dataset_path = f"../../assets/{split_type}.csv"
    if not os.path.exists(dataset_path):
        splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet',
                  'test': 'yelp_review_full/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits[split_type])
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
    return df if n is None else df.iloc[:n, :]


def example_df():
    df = pd.DataFrame(data={"text": ["present. bat. plane."], "label": [1]})
    path_template = "../../assets/example/{label}/{file_name}.tsv"
    process_df(df, path_template)


stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()


def process_df(df: pd.DataFrame, path_template: str):
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        label, text = row["label"], row["text"]
        sentences = tokenize(text)
        save_path = path_template.format(label=label, file_name=f"{idx:06d}")
        label_dir_path = os.path.join(*os.path.split(save_path)[:-1])
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        buffer = ""
        for sent in sentences:
            for word in sent:
                word_stem = stemmer.stem(word)
                word_lemm = lemmer.lemmatize(word)
                buffer += "\t".join([word, word_stem, word_lemm])
                buffer += "\n"
            buffer += "\n"
        with open(save_path, "w") as file:
            file.write(buffer)


def main(split_type: str = "train", n: int | None = 60_000):
    df = load_dataset(split_type, n=n)
    save_path = "../../assets/annotated-corpus-v3/{split_type}/{{label}}/{{file_name}}.tsv"
    save_path = save_path.format(split_type=split_type)
    process_df(df, save_path)


if __name__ == "__main__":
    split_type = "test"
    main(split_type, n=30_000)
