from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_fairness_binary(y_true, y_pred, sensitive_attr):
    results = {}

    # Global metrics
    results["DPD"] = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attr)
    results["EOD"] = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_attr)
    results["Accuracy"] = accuracy_score(y_true, y_pred)
    results["Precision"] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    results["Recall"] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    results["F1"] = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Group metrics
    for group in np.unique(sensitive_attr):
        group_mask = (sensitive_attr == group)
        y_g = y_true[group_mask]
        y_pred_g = y_pred[group_mask]

        tn, fp, fn, tp = confusion_matrix(y_g, y_pred_g, labels=[0, 1]).ravel()
        TPR = tp / (tp + fn + 1e-6)
        FNR = fn / (fn + tp + 1e-6)
        F1g = f1_score(y_g, y_pred_g, zero_division=0)

        results[f"TPR_group{group}"] = TPR
        results[f"FNR_group{group}"] = FNR
        results[f"F1_group{group}"] = F1g

    return results

def evaluate_fairness_multiclass(y_true, y_pred, sensitive_attr, average="macro"):
    classes = np.unique(y_true)
    dpd_list = []
    eod_list = []

    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

        dpd = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_attr)
        eod = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_attr)

        dpd_list.append(dpd)
        eod_list.append(eod)

    if average == "macro":
        dpd_result = np.mean(dpd_list)
        eod_result = np.mean(eod_list)
    else:
        dpd_result = dpd_list
        eod_result = eod_list

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {
        "DPD": dpd_result,
        "EOD": eod_result,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }