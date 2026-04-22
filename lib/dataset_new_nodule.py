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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), normalize])


class ImageDataset_by_list_and_index(data.Dataset):
    def __init__(self, image_list, label_list, name_list, index_list):
        self.data = []
        self.label = []
        self.name = []

        for index__ in index_list:
            num_images = len(image_list[index__])
            for j in range(len(image_list[index__])):

                img = Image.open(image_list[index__][j])
                numpy_img = np.array(img)

                img = Image.fromarray(numpy_img.astype('uint8')).convert('RGB')

                img = TRANSFORM(img)[0, :, :]
                img = torch.unsqueeze(img, 0)

                if j == 0:
                    new_image = img
                else:
                    new_image = torch.cat([img, new_image], dim=0)

            new_image = torch.unsqueeze(new_image, 0)

            self.data.append(new_image)
            self.label.append(label_list[index__])
            self.name.append([num_images, name_list[index__]])

        # postprocess the label
        self.label = torch.Tensor(self.label)
        self.label = self.label.long()

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.name[index]

    def __len__(self):
        return len(self.data)


def load_datalist(datadir):
    np.random.seed(998244353)
    torch.manual_seed(998244353)

    all_list = []
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
        all_list.append([egfr_folder, one_patient_images, 1])

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
        all_list.append([no_egfr_folder, one_patient_images, 0])

    random.shuffle(all_list)

    return image_list, label_list, name_list, all_list


def dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index):
    train_dset = ImageDataset_by_list_and_index(image_list, label_list, name_list, train_index)
    test_dset = ImageDataset_by_list_and_index(image_list, label_list, name_list, val_index)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size, shuffle=True, num_workers=1, )
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size, shuffle=True, num_workers=1, )
    return train_loader, test_loader


def loaderloader(traindir, batch_size):
    image_list, label_list, name_list, all_list = load_datalist(traindir)
    kf = KFold(n_splits=5, shuffle=True)
    i = 0
    for train_index, val_index in kf.split(image_list):
    # for train_index, val_index in kf.split(all_list):
        i += 1
        if i == 1:
            # print(train_index, val_index)
            train_loader1, test_loader1 = dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index)
            # train_loader1, test_loader1 = dset_loader(batch_size, all_list, train_index, val_index)
        if i == 2:
            # print(train_index, val_index)
            train_loader2, test_loader2 = dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index)
            # train_loader2, test_loader2 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 3:
            # print(train_index, val_index)
            train_loader3, test_loader3 = dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index)
            # train_loader3, test_loader3 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 4:
            # print(train_index, val_index)
            train_loader4, test_loader4 = dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index)
            # train_loader4, test_loader4 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)
        if i == 5:
            # print(train_index, val_index)
            train_loader5, test_loader5 = dset_loader(batch_size, image_list, label_list, name_list, train_index, val_index)
            # train_loader5, test_loader5 = dset_loader(batch_size, all_list[1], all_list[2], all_list[0], train_index, val_index)

    return train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, train_loader4, test_loader4, train_loader5, test_loader5


if __name__ == '__main__':

    # alldir = '/disk1/liuzy/dataset/guizhou_dataset/2_split_all_dataset_by_slices_nums/feichuang_all/8_better_adaptive_z90/3_CT_multiply_Mask/dataset_z/1'
    alldir = '/disk1/liuzy/dataset/guizhou_dataset/2_split_all_dataset_by_slices_nums/feichuang_all/9_better_adaptive_z90_left_and_right/3_CT_multiply_Mask/not_splited/0_255/dataset_z/1'

    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, train_loader4, test_loader4, train_loader5, test_loader5 = loaderloader(alldir, 6)
    print('test_loader1')
    for batch_idx, (inputs, targets, names) in enumerate(test_loader1):
        print(names, targets)
        print(inputs.shape)
        break
    # print('test_loader2')
    # for batch_idx, (inputs, targets, names) in enumerate(test_loader2):
    #     print(names, targets)
    # print('test_loader3')
    # for batch_idx, (inputs, targets, names) in enumerate(test_loader3):
    #     print(names, targets)
    # print('test_loader4')
    # for batch_idx, (inputs, targets, names) in enumerate(test_loader4):
    #     print(names, targets)
    # print('test_loader5')
    # for batch_idx, (inputs, targets, names) in enumerate(test_loader5):
    #     print(names, targets)
