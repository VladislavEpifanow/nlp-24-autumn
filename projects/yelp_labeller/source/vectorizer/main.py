from gensim.models import Word2Vec
from tqdm import tqdm
from tqdm.auto import tqdm

from projects.yelp_labeller.source.association_meter.main import apply_steps, steps
from projects.yelp_labeller.source.classifier.main import load_dataset
from projects.yelp_labeller.source.classifier.tokenizer import tokenize


def get_text_tokens(text: str):
    tokens = tokenize(text)
    for sent in tokens:
        cleared_sent = list(filter(lambda x: x, [apply_steps(word, steps) for word in sent]))
        if not cleared_sent:
            continue
        yield cleared_sent
    return


class DFIterator:
    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for idx, row in tqdm(self.df.iterrows(), total=df.shape[0]):
            _, text = row["label"], row["text"]
            for sent in get_text_tokens(text):
                yield sent


if __name__ == "__main__":
    # T = 7h
    split_type = "train"
    model_path = "w2v_model_full_v2"

    df = load_dataset(split_type)
    data_iterator = DFIterator(df)

    seed = 42
    window = 5
    sg = 0  # use CBOW
    cbow_mean = 0
    min_count = 1
    vector_size = 100

    # Обучение модели
    model = Word2Vec(data_iterator, seed=seed, window=window, sg=sg, cbow_mean=cbow_mean, min_count=min_count,
                     vector_size=vector_size)
    model.save(model_path)
