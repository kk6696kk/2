import sys
sys.path.append('/disk1/liuzy/lung cancer project/VGG_egfr4_dense_K_fold')
import torch
import torch.nn.parallel
import torch.cuda
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from numpy import *
import os
from sklearn.model_selection import KFold
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), normalize])

# 数据预处理（标准化）
def process_data(train_x):
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    train_x = (train_x - mean) / std   # print('处理后train_x',train_x)
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
    # print('mean',mean)
    # print('mean',mean.shape)
    # print('std',std)
    print('data normalized!(axis=0)')
    return train_x


class ClinicalDataset(data.Dataset):
    def __init__(self, datas, labels, names, index_list):
        self.data = []
        self.label = []
        self.name = []

        for index__ in index_list:
            self.data.append(torch.Tensor(datas[index__]).float())
            self.label.append(labels[index__])
            self.name.append([index__,names[index__]])  # index_number , patient_name_number

        # postprocess the label
        # self.data = torch.Tensor(self.data).long()
        # self.data = process_data(self.data)
        self.label = torch.Tensor(self.label).long()
        self.name = torch.Tensor(self.name).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.name[index]

    def __len__(self):
        return len(self.data)


def load_clinical_data(datadir):
    np.random.seed(998244353)
    torch.manual_seed(998244353)

    df = pd.read_excel(datadir, sheet_name='173')
    datas = df.values[1:, 3:].astype(float)  # column_0:序号  column_1:检查编号  column_2:0/1(labels) column_3:0/1,man/woman
    labels = df.values[1:, 2].astype(float)
    names = df.values[1:, 1].astype(float)   # astype: torch.tensor required

    datas = process_data(datas)
    # print("all_the_data:\n{0}".format(datas))
    # print("\nclinical_data_shape:", datas.shape)  # (173, 19)
    # print("data_type:", type(datas))  # <class 'numpy.ndarray'>

    # random.shuffle(all_list)

    return datas, labels, names


def dset_loader(batch_size, datas, labels, names, train_index, val_index, number_workers):
    train_dset = ClinicalDataset(datas, labels, names, train_index)
    test_dset = ClinicalDataset(datas, labels, names, val_index)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size, shuffle=True, num_workers=number_workers, )
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size, shuffle=True, num_workers=number_workers, )
    return train_loader, test_loader


def loaderloader(traindir, batch_size, number_workers):
    data, label, name = load_clinical_data(traindir)
    kf = KFold(n_splits=5, shuffle=True)
    i = 0
    for train_index, val_index in kf.split(data):
    # for train_index, val_index in kf.split(all_list):
        i += 1
        if i == 1:
            # print(train_index, val_index)
            train_loader1, test_loader1 = dset_loader(batch_size, data, label, name, train_index, val_index, number_workers)
            # train_loader1, test_loader1 = dset_loader(batch_size, all_list, train_index, val_index)
        if i == 2:
            # print(train_index, val_index)
            train_loader2, test_loader2 = dset_loader(batch_size, data, label, name, train_index, val_index, number_workers)
            # train_loader2, test_loader2 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 3:
            # print(train_index, val_index)
            train_loader3, test_loader3 = dset_loader(batch_size, data, label, name, train_index, val_index, number_workers)
            # train_loader3, test_loader3 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 4:
            # print(train_index, val_index)
            train_loader4, test_loader4 = dset_loader(batch_size, data, label, name, train_index, val_index, number_workers)
            # train_loader4, test_loader4 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 5:
            # print(train_index, val_index)
            train_loader5, test_loader5 = dset_loader(batch_size, data, label, name, train_index, val_index, number_workers)
            # train_loader5, test_loader5 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)

    return train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, train_loader4, test_loader4, train_loader5, test_loader5


if __name__ == '__main__':

    alldir = '/data2/lwy/dataset/clinical_data.xlsx'
    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, \
                    train_loader4, test_loader4, train_loader5, test_loader5 = loaderloader(alldir, 4, 1)

    print('\ntest_loader1')
    for batch_idx, (inputs, targets, names) in enumerate(test_loader1):  # data dim is 19
        print('names:',names)
        print('targets:',targets)
        print('inputs.shape:',inputs)
        break

