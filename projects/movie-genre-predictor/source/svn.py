import time
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import os

# %%
df = pd.read_csv("../assets/annotated_corpus/train-embeddings.tsv", sep="\t", header=None, index_col=False)

# %%
df_test = pd.read_csv("../assets/annotated_corpus/test-embeddings.tsv", sep="\t", header=None, index_col=False)

# %%
df.info()

# %%
df["target"] = df[0].str.rsplit("_", n=1, expand=True)[0]
df_test["target"] = df_test[0].str.rsplit("_", n=1, expand=True)[0]
df.head()

# %%
label_encoder = LabelEncoder()
df["target_enc"] = label_encoder.fit_transform(df["target"])
df_test["target_enc"] = label_encoder.fit_transform(df_test["target"])
df[["target", "target_enc"]]

# %%
df["target"].unique(), df["target_enc"].unique()

# %%
def confusion_matrix(true, pred):
    classes = set(true + pred)
    num_classes = len(classes)
    mat = np.zeros((num_classes, num_classes))
    n = max(len(true), len(pred))
    for i in range(num_classes):
        for j in range(num_classes):
            for k in range(n):
                if true[k] == i:
                    if pred[k] == j:
                        mat[i][j] = mat[i][j] + 1
    return mat

# %%
def get_precision_recall_fscore_accuracy(cm, beta=2.0):
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    numerator = (1 + math.pow(beta, 2)) * recall * precision
    denominator = (math.pow(beta, 2) * precision) + recall

    fscore = numerator / denominator

    accuracy = true_pos / np.sum(cm, axis=1)

    return precision, recall, fscore, accuracy

# %%
params = [
    { 
        "kernel": ["linear"],
        # "gamma": ["scale", "auto"],
        "max_iter": [10, 100, 1000]
    },
    { 
        "kernel": ["poly"],
        "degree": [3],
        "gamma": ["scale", "auto"],
        "max_iter": [10, 100, 1000]
        # "class_weight": [None, "balanced"]
    },
    {
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
        "max_iter": [10, 100, 1000]
        # "class_weight": [None, "balanced"]
    },
    {
        "kernel": ["sigmoid"],
        "max_iter": [10, 100, 1000],
        "gamma": ["scale", "auto"]
    }
]

param_grid = ParameterGrid(params)

# %%
X = df[df.columns.difference([0, 'target', 'target_enc'])]
y = df["target_enc"]

X_test = df_test[df_test.columns.difference([0, 'target', 'target_enc'])]
y_test = df_test["target_enc"]

# %%
def grid_search(X, y, param_grid):
    metrics = {
        "accuracy": dict(),
        "precision": dict(),
        "recall": dict(),
        "fscore" : dict(),
        "exec_time": dict()
    }

    model_params = dict()
    
    metrics_names = ["accuracy", "precision", "recall", "fscore", "exec_time"]
    for i, param in enumerate(param_grid):
        clf = SVC(random_state=42, **param)
        
        start_time = time.time()
        clf.fit(X, y)
        exec_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
    
        cm = confusion_matrix(y_test.tolist(), y_pred.tolist())
        pr, rec, fscore, acc = get_precision_recall_fscore_accuracy(cm)
    
        print(f"Model version №{i + 1}")
        print("params", param)
        for metr, name in zip([acc, pr, rec, fscore, exec_time], metrics_names):
            metrics[name][f"model_{i + 1}"] = metr
            # print(name, np.mean(metr))

        model_params[f"model_{i + 1}"] = param

    return metrics, model_params
    

# %%
def find_best_model_by_metrics(metric_model, metrics_names):
    for name in metrics_names:
        k, v = max(metric_model[name].items(), key=lambda x: np.mean(x[1]))
        print(f"Metric {name}: model {k} with mean value {np.mean(v)}")

# %%
metrics, model_params_1 = grid_search(X, y, param_grid)

# %%
find_best_model_by_metrics(metrics, ["accuracy", "precision", "recall", "fscore"])

# %%
best_model = "model_6"
best_model_metrics = { metric_name: val for metric_name, d in metrics.items() for model_name, val in d.items() if model_name == best_model }


# %%
best_model_metrics

# %%
X_log = X.copy().apply(np.log).fillna(0.0)
X_sin = X.copy().apply(np.sin)

# %%
metrics_names = ["accuracy", "precision", "recall", "fscore", "exec_time"]


# %%
def fit_predict(X, y, model_params):
    clf = SVC(**model_params_1[best_model])
    start_time = time.time()
    clf.fit(X, y)
    exec_time = time.time() - start_time
    
    y_pred = clf.predict(X)
    cm = confusion_matrix(y.tolist(), y_pred.tolist())
    pr, rec, fscore, acc = get_precision_recall_fscore_accuracy(cm)
    
    metrics = dict()
    for metr, name in zip([acc, pr, rec, fscore, exec_time], metrics_names):
        metrics[name] = metr
        print(name, np.mean(metr))

    return metrics

# %%
log_model_metrics = fit_predict(X_log, y, model_params_1[best_model])


# %%
sin_model_metrics = fit_predict(X_sin, y, model_params_1[best_model])


