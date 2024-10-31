import numpy as np

def calculate_metrics(y_true: np.array, y_pred: np.array):
    num_classes = len(np.unique(y_true))

    precision = 0
    recall = 0
    f1_score = 0

    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision += tp / (tp + fp) if (tp + fp) > 0 else 0
        recall += tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score += 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    precision /= num_classes
    recall /= num_classes
    f1_score /= num_classes

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return precision, recall, f1_score, accuracy