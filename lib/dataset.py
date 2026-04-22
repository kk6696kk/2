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
import cv2
import tensorflow as tf
import random
from scipy.interpolate import griddata
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
# TRANSFORM = transforms.Compose([transforms.ToTensor(), normalize])
TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), normalize])
# TEST_TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), normalize])


class ImageDataset_train(data.Dataset):
    def __init__(self, image_list, label_list, name_list):
        self.data = []
        self.label = []
        self.name = []
        for i in range(len(image_list)):
            num_images = len(image_list[i])
            for j in range(len(image_list[i])):
                image_list[i].sort()

                img = Image.open(image_list[i][j])
                img = np.array(img)
                img = Image.fromarray(img.astype('uint8')).convert('RGB')
                img = TRANSFORM(img)[0, :, :]
                img = torch.unsqueeze(img, 0)

                if j == 0:
                    new_image = img
                else:
                    new_image = torch.cat([img, new_image], dim=0)

            # print('new_image_shape:', new_image.shape)
            new_image = torch.unsqueeze(new_image, 0)

            self.data.append(new_image)
            self.label.append(label_list[i])
            self.name.append(['ori__', num_images, name_list[i]])

        for i in range(len(image_list)):
            num_images = len(image_list[i])

            key, value = select_data_augumentation_key_value(image_list[i][0])
            for j in range(len(image_list[i])):
                image_list[i].sort()

                img = Image.open(image_list[i][j])
                img = np.array(img)

                agued_img = data_augumentation(key, value, img)

                agued_img = Image.fromarray(agued_img.astype('uint8')).convert('RGB')
                agued_img = TRANSFORM(agued_img)[0, :, :]
                agued_img = torch.unsqueeze(agued_img, 0)
                if j == 0:
                    new_agued_img = agued_img
                else:
                    new_agued_img = torch.cat([agued_img, new_agued_img], dim=0)

            new_agued_img = torch.unsqueeze(new_agued_img, 0)

            self.data.append(new_agued_img)
            self.label.append(label_list[i])
            self.name.append(['agued', num_images, name_list[i]])

        # postprocess the label
        self.label = torch.Tensor(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.name[index]

    def __len__(self):
        return len(self.data)


class ImageDataset_test(data.Dataset):
    def __init__(self, image_list, label_list, name_list):
        self.data = []
        self.label = []
        self.name = []
        for i in range(len(image_list)):
            num_images = len(image_list[i])
            for j in range(len(image_list[i])):
                image_list[i].sort()

                img = Image.open(image_list[i][j])
                img = np.array(img)

                img = Image.fromarray(img.astype('uint8')).convert('RGB')

                img = TRANSFORM(img)[0, :, :]
                img = torch.unsqueeze(img, 0)

                if j == 0:
                    new_image = img
                else:
                    new_image = torch.cat([img, new_image], dim=0)

            # print('new_image_shape:', new_image.shape)
            new_image = torch.unsqueeze(new_image, 0)

            self.data.append(new_image)
            self.label.append(label_list[i])
            self.name.append([num_images, name_list[i]])

        # postprocess the label
        self.label = torch.Tensor(self.label)
        self.label = self.label.long()

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.name[index]

    def __len__(self):
        return len(self.data)


def load_dataset(datadir, split):
    np.random.seed(998244353)
    torch.manual_seed(998244353)
    image_list = []
    label_list = []
    name_list = []

    egfr_dir = os.path.join(datadir, 'egfr')
    egfr_folders = os.listdir(egfr_dir)
    for egfr_folder in egfr_folders:
        egfr_files = os.listdir(os.path.join(egfr_dir, egfr_folder))
        egfr_files.sort()
        one_patient_images = []
        for egfr_file in egfr_files:
            one_patient_images.append(os.path.join(egfr_dir, egfr_folder, egfr_file))
        image_list.append(one_patient_images)
        label_list.append(1)
        name_list.append(egfr_folder)

    no_egfr_dir = os.path.join(datadir, 'egfr_no')
    no_egfr_folders = os.listdir(no_egfr_dir)
    for no_egfr_folder in no_egfr_folders:
        no_egfr_files = os.listdir(os.path.join(no_egfr_dir, no_egfr_folder))
        no_egfr_files.sort()
        one_patient_images = []
        for no_egfr_file in no_egfr_files:
            one_patient_images.append(os.path.join(no_egfr_dir, no_egfr_folder, no_egfr_file))
        image_list.append(one_patient_images)
        label_list.append(0)
        name_list.append(no_egfr_folder)

    if split == 'train':
        dset = ImageDataset_train(image_list, label_list, name_list)
    else:
        dset = ImageDataset_test(image_list, label_list, name_list)

    return dset


def data_augumentation(key, value, ori_img):
    if key == '':
        new_img = ori_img
    if key == 'flip_lr':
        # flip lift and right
        new_img = np.fliplr(ori_img)

    if key == 'flip_uds':
        # flip up and dowm
        new_img = np.flipud(ori_img)

    if key == 'rotate_180':
        new_img = ori_img[::-1, ...][:, ::-1]

    if key == 'rotate_90':
        new_img = np.transpose(ori_img, axes=(1, 0))[::-1, ...]

    if key == 'random_crop':
        x, y = ori_img.shape
        biu = value
        if x <= y:
            new_img = ori_img[0:x - 1, biu:biu + x]
        else:
            new_img = ori_img[biu:biu + y, 0:y - 1]

    return new_img


def select_data_augumentation_key_value(img):
    list_key_value = ['flip_lr', 'flip_uds', 'rotate_180', 'rotate_90', 'random_crop']
    # list_key_value = ['random_crop']
    key = random.choice(list_key_value)
    if key == 'random_crop':
        img = Image.open(img)
        img = np.array(img)
        x, y = img.shape
        if x <= y:
            value = np.random.randint(0, y - x)
        else:
            value = np.random.randint(0, x - y)
    else:
        value = 0

    return key, value


if __name__ == '__main__':

    traindir = '/disk1/liuzy/dataset/guizhou_dataset/2_split_all_dataset_by_slices_nums/feichuang_all/8_better_adaptive_z90/3_CT_multiply_Mask/dataset_z/2_6vs1/val'

    train_dset = load_dataset(traindir, 'train')

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=6, shuffle=True, num_workers=1,)

    for batch_idx, (inputs, targets, names) in enumerate(train_loader):

        print(targets, names)
        print(inputs.size())

        # input = inputs[0, 0, :, :, :]
        # for i in range(input.shape[0]):
        #     input_ = input[i, :, :]
        #     input_ = np.array(input_) * 255
        #     print(input_.shape)
        #     cv2.imwrite('/disk1/liuzy/dataset/guizhou_dataset/2_split_all_dataset_by_slices_nums/feichuang_all/6_333/4_datasets/all/1_demo/test/' + str(i).zfill(3)+'.png',
        #                 input_)

        # for input in inputs:
        #     # input = torch.squeeze(input, 0)
        #     print(input.size())
        #
        break
