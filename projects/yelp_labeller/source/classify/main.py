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
    df = load_dataset(split_type=split_type)
    return df["label"].to_numpy() if n is None else df["label"][:n].to_numpy()


def load_emb_dataset(split: str, n = None):
    emb_path = rf"..\..\assets\{split}_embeddings.tsv"
    doc_ids_dict, x = read_emb(emb_path, n=n)
    print(x.shape)

    y = get_doc_label(split, n=len(doc_ids_dict))
    print(y.shape)
    return x, y


if __name__ == "__main__":
    random_state = 42
    use_pca = True
    n_components = 10
    runs_file_path = "runs_log.txt" if not use_pca else f"runs_log_pca_{n_components}.txt"

    x_train, y_train = load_emb_dataset('train', n=30_000)
    x_test, y_test = load_emb_dataset('test', n=10_000)

    if use_pca:
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

    print("\n".join(runs_data))
    with open(runs_file_path, "w+") as file:
        for run in runs_data:
            file.write(json.dumps(run)+"\n")
