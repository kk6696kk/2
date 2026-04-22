from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
import os


def ROC_AUC(y,pred):
    '''
    input:
        y:the label of the data
        pred:the model output of the data
    output:
        fpr:x of the ROC figure
        tpr:y of the ROC figure
        auc_score:area under the ROC
    '''
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    auc_socre = auc(fpr, tpr)
    return fpr, tpr, auc_socre

def Five_FOld_MeanAUC(path, model_dic):
    # inital
    mean_fpr = np.linspace(0,1,1000)
    tprs = []
    aucs = []
    path = path
    model_dic = model_dic

    for i in range(1,6):
        # load the target and the pred
        target = np.load(os.path.join(path, model_dic, str(i), "raw_auc_curve_gt_all.npy"))
        pred = np.load(os.path.join(path, model_dic, str(i), "raw_auc_curve_prob_all.npy"))
        # calculate the every fold auc
        fpr, tpr, auc_score = ROC_AUC(target, pred)
        # append the tprs to calculate the mean_tpr
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc_score)

    # calculate the mean_tpr
    mean_tpr = np.mean(tprs,axis=0)
    mean_tpr[-1] = 1.0
    # calculate the mean_auc
    mean_auc = auc(mean_fpr,mean_tpr)
    return mean_fpr, mean_tpr, mean_auc




if __name__ == "__main__":
    path = "/data2/lwy/projevt/Lung/9.23/VGG_egfr9_Dialated_AC_in_Dense_K_fold_new/output/5_23_output_ROC"
    ##########
    model_dic = "model1_densenet_11_12_ROI_contrast"
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path,model_dic)
    plt.plot(mean_fpr,mean_tpr, color='g', lw=2,alpha=.8)
    ##########
    model_dic = 'model3_7_15_SVM_64_model_noNormalize'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8)
    ##########
    model_dic = 'model12_7_14_VGG16_64_model'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='c', lw=2, alpha=.8)
    ##########
    model_dic = "model15_11_12_SEResnet2D_64_model"
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='m', lw=2, alpha=.8)
    ##########
    model_dic = "model9_11_12_Resnet2D_151_transferlearning_model"
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='y', lw=2, alpha=.8)
    ##########
    model_dic = 'model9_11_12_Resnet_3D_41_transferlearning_model'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='k', lw=2, alpha=.8)
    ##########
    model_dic = 'model19_7_11_Resnet1013D_64_model_mixup'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, lw=2, alpha=.8)
    ##########
    model_dic = 'model16_7_14_Densenet3D_128_model_noNormalize'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr,lw=2, alpha=.8)
    ##########
    model_dic = 'model20_7_14_Densenet3D_48_model_mixup_noNormalize'
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, lw=2, alpha=.8)
    ##########
    model_dic = "model_Denseformer_output_96"
    mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    plt.plot(mean_fpr, mean_tpr, color='r', label=r'Denseformer', lw=2, alpha=.8)
    # ##########
    # model_dic = "12_15_nopretrained_layer4_96_noDialated_Tran_Dialated_AC_in_DenseNet9_(CT2_second_hu_0_255)_(utils_acc_auc)_5fold_bs4_1"
    # mean_fpr, mean_tpr, mean_auc = Five_FOld_MeanAUC(path, model_dic)
    # plt.plot(mean_fpr, mean_tpr, color='r', label=r'Denseformer(3D)(area=%0.3f)' % mean_auc, lw=2, alpha=.8)

    # universal
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig('ROC.png')
    plt.show()


