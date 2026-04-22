from __future__ import print_function
import sys
sys.path.append('/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold')
import torch
import torch.nn.parallel
from ablation_network.Dialated_AC_in_DenseNet9 import DenseNet
from visualize import dataset
import os
import cv2
import numpy as np
import torch.nn as nn
from lib.eval_acc_auc import AUC_score
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D


def load_checkpoint(model, checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(model_CKPT['state_dict'])
    print('succeed loading checkpoint!')

    return model


def p_all_CTs(Nodule_dir, axe):
    Nodule_folders = os.listdir(Nodule_dir)
    res = {}
    for Nodule_folder in Nodule_folders:
        indexes = os.listdir(os.path.join(Nodule_dir, Nodule_folder, axe))
        indexes.sort()
        selected_index = '000.png'
        max_count = 0
        for index in indexes:
            image = cv2.imread(os.path.join(Nodule_dir, Nodule_folder, axe, index))
            array = image[:, :, 0]
            a = np.count_nonzero(array)
            if a > max_count:
                max_count = a
                selected_index = index
        res[Nodule_folder] = selected_index
    return res


def heatmap_image(heatmap_3d, image, idx, label, save, name):
    img = cv2.imread(os.path.join(image, idx))
    idx_num = int(idx.split('.')[0])
    # heatmap_test = heatmap_3d[idx_num, :, :]
    heatmap_test = heatmap_3d[111-idx_num, :, :]
    heatmap_test = np.uint8(255 * heatmap_test)
    heatmap_test = cv2.applyColorMap(heatmap_test, cv2.COLORMAP_JET)

    superimposed_img_test = heatmap_test * 0.3 + img
    superimposed_img_test = np.clip(superimposed_img_test, 0, 255)
    # superimposed_img_test[img == 0] = 0
    superimposed_img_test = superimposed_img_test.astype(np.uint8)

    # Save
    if label[0]==label[1]:
        if label[1]==1:
            save_path = os.path.join(save, 'correct', 'egfr')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, name+'_'+idx), superimposed_img_test)
        elif label[1]==0:
            save_path = os.path.join(save, 'correct', 'egfr_no')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, name+'_'+idx), superimposed_img_test)


def test(test_loader, slices, save_visual):
    pred_score_all, gt_all = [], []
    for batch_idx, (input, targets, paths) in enumerate(test_loader):
        targets = torch.autograd.Variable(targets)
        inputs = torch.autograd.Variable(input)
        with torch.no_grad():
            outputs = model(inputs)
            features = model.feature
        outputs = nn.Softmax(dim=1)(outputs)

        # 可视化
        for i, feature in enumerate(features):
            # print(inputs.shape)
            heatmap = feature.detach().numpy()
            heatmap = np.mean(heatmap, axis=0)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap = scipy.ndimage.zoom(heatmap, 112 / 6, order=2)

            # Output
            name = paths[i].split('/')[-2]
            print(name)
            network_output = outputs[i].data.cpu().numpy()
            network_res = 1 if max(network_output)==network_output[1] else 0
            gt_res = targets[i].data.cpu().numpy()

            heatmap_image(heatmap, paths[i], slices[name], [network_res, gt_res], save_visual, name)

        # Auc
        outputs_numpy = outputs.data.cpu().numpy()
        targets_numpy = targets.data.cpu().numpy()
        for i in range(len(targets_numpy)):
            pred_score_all.append(outputs_numpy[i][-1])
            gt_all.append(targets_numpy[i])

    auc = AUC_score(np.array(gt_all), np.array(pred_score_all))

    return auc


if __name__=='__main__':
    # data_dir = '/disk1/liuzy/dataset/guizhou_dataset/all_new/Dataset_112_112_112/CT/dataset1_second_hu/0_255'
    # data_dir = '/disk1/liuzy/dataset/guizhou_dataset/all_new/Dataset_112_112_112/CT/dataset1_second_hu/0_255'
    data_dir = '/disk1/liuzy/dataset/guizhou_dataset/all_new/Dataset_112_112_112/CT/dataset1_second_hu_x_and_y/0_255'
    checkpoint_PATH = '/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/output_ablation/Dialated_AC_in_DenseNet9_(CT_112_112_112_second_hu_0_255)_(utils_acc_auc)_5fold_bs4_1/4/model_best.pth.tar'
    save_dir = '/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/visualize/res/112_112_112/test5_all_select_one/x'

    # dataloader
    all_loader = dataset.initialize_datasets(data_dir, 'x')

    # slice_selected
    slices = p_all_CTs('/disk1/liuzy/dataset/guizhou_dataset/all_new/Visualization/112_112_112/Nodule_mask_x_y', 'x')

    # model
    device = torch.device('cpu')
    model = DenseNet().to(device)
    model_ = load_checkpoint(model, checkpoint_PATH)

    # test
    auc = test(all_loader, slices, save_dir)
    print(auc)