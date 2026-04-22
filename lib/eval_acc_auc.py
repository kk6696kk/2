from __future__ import print_function, absolute_import
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


__all__ = ['accuracy']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AUC_score(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score


def AUC_score_new(label_all, prob_all):
    return roc_auc_score(label_all, prob_all)


if __name__ == '__main__':
    # input = torch.randn(3, 2, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(2)
    # print(input)
    # print('target', target)
    label_all = [1, 0, 1, 0, 1, 0, 1, 0]
    prob_all = [1, 0, 1, 1, 1, 0, 1, 0]
    res1 = AUC_score(label_all, prob_all)
    res2 = AUC_score_new(label_all, prob_all)
    print(res1, res2)
