import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from scipy import interp
import numpy as np


def draw_roc(res_dir):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    folders = os.listdir(res_dir)
    for folder in folders:
        if folder == 'res.txt':
            continue
        gt_path = os.path.join(res_dir, folder, 'raw_auc_curve_gt_all.npy')
        prob_path = os.path.join(res_dir, folder, 'raw_auc_curve_prob_all.npy')
        gt_list = np.load(gt_path)
        prob_list = np.load(prob_path)

        fpr, tpr, thresholds = roc_curve(gt_list, prob_list)
        auc_res = auc(fpr, tpr)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc_res)

        plt.plot(fpr, tpr, lw=1.5, label="W-Y, AUC=%.3f)" % auc_res)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='r', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel("False positive rate", fontsize=15)
    plt.ylabel("True positive rate", fontsize=15)
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()


if __name__=='__main__':
    res_dir = '/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/output_ablation/Dialated_AC_in_DenseNet9_(CT2_second_hu_0_255)_(utils_acc_auc)_5fold_bs6_1'
    draw_roc(res_dir)
