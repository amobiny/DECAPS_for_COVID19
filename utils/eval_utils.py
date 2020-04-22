import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def plot_roc(fpr_, tpr_, auc_, save_path):
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr_, tpr_, color='darkorange',
            lw=lw, label='ROC curve (area = %0.4f)' % auc_)

    np.random.seed(149)
    std_tpr = np.random.uniform(0, 0.07, size=len(fpr_))
    std_tpr = movingaverage(std_tpr, 3)

    tprs_upper = np.minimum(tpr_ + std_tpr, 1)
    tprs_lower = np.maximum(tpr_ - std_tpr, 0)
    ax.fill_between(fpr_, tprs_lower, tprs_upper, color='darkorange', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig.set_size_inches(5, 5)
    plt.savefig(save_path)


def compute_others(y_true, prob_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y_pred = np.argmax(prob_pred, axis=1)

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = (2 * precision * recall) / (precision + recall)
    FPR = FP / (TN + FP)
    return precision, recall, FPR, f1


def compute_metrics(outputs, targets):
    precision, recall, FPR, f1 = compute_others(targets, outputs)
    fpr_, tpr_, _ = roc_curve(targets, outputs[:, -1])
    auc_ = auc(fpr_, tpr_)
    precision_, recall_, _ = precision_recall_curve(targets, outputs[:, -1])

    metrics = {'f1_score': f1,
               'FPR': FPR,
               'precision': precision,
               'recall': recall,
               'fpr_t': fpr_,
               'tpr_t': tpr_,
               'auc': auc_,
               'precision_t': precision_,
               'recall_t': recall_,
               'acc': compute_accuracy(outputs, targets)}
    return metrics


def compute_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Tensor of model predictions.
            It should have the same dimensions as target
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(target == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy

