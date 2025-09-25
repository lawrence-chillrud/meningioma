import numpy as np
import torch

def all_metrics(y_true, y_pred, average='macro'):
    # Ensure y_pred is in the same format as y_true (e.g., class indices for classification)
    if y_pred.ndim > 1:  # If y_pred is probabilities or logits
        y_pred = y_pred.argmax(dim=1)

    # Compute confusion matrix components
    num_classes = len(torch.unique(y_true))
    cnf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=y_true.device)
    for t, p in zip(y_true, y_pred):
        cnf_matrix[t.long(), p.long()] += 1

    FP = cnf_matrix.sum(dim=0) - torch.diag(cnf_matrix)
    FN = cnf_matrix.sum(dim=1) - torch.diag(cnf_matrix)
    TP = torch.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Avoid division by zero
    epsilon = 1e-7
    FP = FP + epsilon
    FN = FN + epsilon
    TP = TP + epsilon
    TN = TN + epsilon

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # Balanced accuracy
    BACC = 0.5 * (TP / (TP + FN) + TN / (TN + FP))

    metrics = {
        "TPR": TPR.cpu().numpy(),
        "TNR": TNR.cpu().numpy(),
        "PPV": PPV.cpu().numpy(),
        "NPV": NPV.cpu().numpy(),
        "FPR": FPR.cpu().numpy(),
        "FNR": FNR.cpu().numpy(),
        "FDR": FDR.cpu().numpy(),
        "ACC": ACC.cpu().numpy(),
        "BACC": BACC.cpu().numpy()
    }

    def calculate_average(metric, average_type):
        if average_type == "macro":
            return metric.mean().item()
        elif average_type == "micro":
            total_tp = TP.sum()
            total_fn = FN.sum()
            total_tn = TN.sum()
            total_fp = FP.sum()
            if metric is TPR:
                return (total_tp / (total_tp + total_fn)).item()
            elif metric is TNR:
                return (total_tn / (total_tn + total_fp)).item()
            elif metric is PPV:
                return (total_tp / (total_tp + total_fp)).item()
            elif metric is NPV:
                return (total_tn / (total_tn + total_fn)).item()
        elif average_type == "weighted":
            weights = (TP + FN) / (TP + FP + FN + TN)
            return (metric * weights).sum().item()
        else:
            raise ValueError("Invalid average_type. Choose from 'macro', 'micro', or 'weighted'.")

    for m in metrics:
        metrics[m] = calculate_average(metrics[m], average_type=average)
    return metrics

def balanced_accuracy(y_true, y_pred, multiclass=False):
    if multiclass:
        y_pred = y_pred.argmax(dim=1)
        class_accuracies = []
        for cls in torch.unique(y_true):
            tp = ((y_true == cls) & (y_pred == cls)).sum()
            fn = ((y_true == cls) & (y_pred != cls)).sum()
            tn = ((y_true != cls) & (y_pred != cls)).sum()
            fp = ((y_true != cls) & (y_pred == cls)).sum()
            class_accuracies.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
        return torch.mean(torch.tensor(class_accuracies))
    else:
        y_pred = (y_pred > 0.5).int()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))

def true_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn)

def false_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return fp / (fp + tn)

def false_discovery_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return fp / (fp + tp)

def tpr_at_fpr(y_true, y_pred, target_fpr=0.1):
    thresholds = np.linspace(0, 1, 100)
    fprs = []

    for t in thresholds:
        fpr = false_positive_rate(y_true, y_pred, threshold=t)
        fprs.append(fpr)

    # Replace NaNs with inf to prevent invalid argmin results
    fprs = np.where(np.isnan(fprs), np.inf, fprs)

    # Compute the closest index and ensure it's in valid bounds
    th_idx = np.argmin(np.abs(fprs - target_fpr))
    th_idx = np.clip(th_idx, 0, len(thresholds) - 1)

    return true_positive_rate(y_true, y_pred, threshold=thresholds[th_idx])