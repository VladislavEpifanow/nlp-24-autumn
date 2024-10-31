import json
import time

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from tqdm import tqdm

from projects.yelp_labeller.source.classifier.main import load_dataset
from projects.yelp_labeller.source.classify.metrics import calculate_metrics


def read_emb(file_path: str, n: int | None):
    doc_ids = []
    doc_emb = []
    cnt = 0
    with open(file_path, "r") as file:
        for line in file.readlines():
            if n is not None and cnt >= n:
                break
            doc_id, *emb = line.split("\t")
            doc_ids.append(doc_id)
            doc_emb.append(emb)
            cnt += 1
    doc_ids_dict = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    return doc_ids_dict, np.asarray(doc_emb)


def get_doc_label(split_type: str, n= None):
    df = load_dataset(split_type=split_type, n=n)
    return df["label"].to_numpy()


def load_emb_dataset(emb_path, split: str, n = None):
    my_emb_path = emb_path.format(split=split, n=n)
    doc_ids_dict, x = read_emb(my_emb_path, n=n)
    print(x.shape)

    y = get_doc_label(split, n=n)
    print(y.shape)
    return x, y


if __name__ == "__main__":
    random_state = 42
    n_components = 0
    # runs_file_path = "runs_log_hug.txt" if not n_components else f"runs_log_pca_{n_components}_hug.txt"
    runs_file_path = "runs_log_lib.txt" if not n_components else f"runs_log_pca_{n_components}_lib.txt"

    # dataset_path = r"..\..\assets\{split}_embeddings_hug_{n}.tsv"
    dataset_path = r"..\..\assets\{split}_embeddings_lib_{n}.tsv"
    train_size, test_size = 10_000, 5_000

    x_train, y_train = load_emb_dataset(dataset_path, 'train', n=train_size)
    x_test, y_test = load_emb_dataset(dataset_path, 'test', n=test_size)

    if n_components:
        pca = PCA(n_components=n_components)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    runs_data = []

    for kernel in tqdm(kernels):
        data = {}
        runs_data.append(data)

        data["params"] = dict(kernel=kernel, random_state=random_state)
        clf = svm.SVC(**data["params"])

        start_time = time.time()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_train_pred = clf.predict(x_train)
        run_time = time.time() - start_time


        precision, recall, f1_score, accuracy = calculate_metrics(y_test, y_pred)
        data["time"] = run_time
        data["precision"] = precision
        data["recall"] = recall
        data["f1_score"] = f1_score
        data["accuracy"] = accuracy

        *_, accuracy = calculate_metrics(y_train_pred, y_train)
        data["train_accuracy"] = accuracy

    with open(runs_file_path, "w+") as file:
        for run in runs_data:
            file.write(json.dumps(run)+"\n")
