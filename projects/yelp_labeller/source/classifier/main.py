import os.path

import pandas as pd

from nltk import SnowballStemmer, WordNetLemmatizer
import nltk

from projects.yelp_labeller.source.classifier.tokenizer import tokenize

# nltk.download('wordnet')

def load_dataset(split_type = "train"):
    assert split_type == "train" or split_type == "test"
    dataset_path = f"../../assets/{split_type}.csv"
    if not os.path.exists(dataset_path):
        splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits[split_type])
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
    return df

def example_df():
    df = pd.DataFrame(data={"text":["present. bat. plane."], "label": [1]})
    path_template = "../../assets/example/{label}/{file_name}.tsv"
    process_df(df, path_template)


stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()

def process_df(df: pd.DataFrame, path_template: str):
    for idx, row in df.iterrows():
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
                buffer+="\t".join([word, word_stem, word_lemm])
                buffer+="\n"
            buffer+="\n"
        with open(save_path, "w") as file:
            file.write(buffer)

def main():
    split_type = "test"
    df = load_dataset(split_type)
    save_path =  "../../assets/annotated-corpus/{split_type}/{{label}}/{{file_name}}.tsv"
    save_path = save_path.format(split_type=split_type)
    process_df(df, save_path)


if __name__=="__main__":
    example_df()
