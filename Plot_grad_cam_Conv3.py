######################################################################
# 1.Import the lib
######################################################################
print('\n' + '1.Import the lib')
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import numpy as np
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt

import time
from network.Denseformer_no_Dialated import DenseNet
from lib.Pretrain_freeze import load_pretrained_encoder_parameter_to_CTmodel
from lib.dataset_new_CT import load_datalist
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import SimpleITK as sitk
from lib.dataset_new_CT import ImageDataset_by_list_and_index


start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
######################################################################
# 2.Load the model
######################################################################
print('\n' + '2.Load the model:')
model = DenseNet(
        stand_dim=96,
        transformer_layers=1,
        transformer_heads=1,
        dropout_p=0
    )
checkpoint_path = '/data2/lwy/projevt/Lung/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/output_173_90_112_112/' \
                  '5_21_nopretrained_96_noDialated_Tran_Dialated_AC_in_DenseNet9_(CT2_second_hu_0_255)_(utils_acc_auc)_5fold_bs4_1/1/model_best.pth.tar'

# checkpoint = torch.load(checkpoint_path)
# load the pretrained parameter
model = load_pretrained_encoder_parameter_to_CTmodel(model, checkpoint_path)
model.eval()
# print the model
print(model)

# model.cuda()


######################################################################
# 3.Select the target layer
######################################################################
print('\n' + '3.Select the target layer:')
# select the conv layer
conv_layer_num = 3
target_layer_list = [model.block[0].block[6].Net[0],
                    model.block[1].block[6].Net[0],
                    model.block[2].block[6].Net[0],
                    model.block[3].block[6].Net[0]]
target_layer = [target_layer_list[conv_layer_num]]

# final layer(t-SNE)
# model.FC



######################################################################
# 4.Input the CT images
######################################################################
print('\n' + '4.Input the CT images:')
# load the img and trans to tensor
CT_path = '/data2/lwy/dataset/0_255'
image_list, label_list, name_list, all_list = load_datalist(CT_path)
# dataset pre
index_list = range(173)
data = []
data_np = []
label = []
name = []
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
TRANSFORM = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), normalize])
# dataset
for index__ in tqdm(index_list):
    num_images = len(image_list[index__])
    for j in range(len(image_list[index__])):

        img_cv2 = cv2.imread(image_list[index__][j])  # numpy 112*112*3
        # # show the img_cv2
        # cv2.imshow('img_cv2', img_cv2)
        # cv2.waitKey(0)
        img_cv2 = np.expand_dims(img_cv2, axis=0)

        img = Image.open(image_list[index__][j])   # numpy 112*112
        numpy_img = np.array(img)
        # # show the img
        # cv2.imshow('numpy_img', numpy_img)
        # cv2.waitKey(0)

        img = Image.fromarray(numpy_img.astype('uint8')).convert('RGB')

        img = TRANSFORM(img)[0, :, :]
        img = torch.unsqueeze(img, 0)

        if j == 0:
            new_image = img
            new_image_cv2 = img_cv2
        else:
            new_image = torch.cat([img, new_image], dim=0)  # !!!!
            new_image_cv2 = np.concatenate([img_cv2, new_image_cv2], axis=0)

    new_image = torch.unsqueeze(new_image, 0)
    new_image_cv2 = np.expand_dims(new_image_cv2, axis=0)

    data.append(new_image)
    data_np.append(new_image_cv2)
    label.append(label_list[index__])
    name.append(name_list[index__])

# postprocess the label
# data = torch.Tensor(data)
label = torch.Tensor(label)
label = label.long()

# # dataset
# test_dset = ImageDataset_by_list_and_index(image_list, label_list, name_list, range(173), mixup=False)
# # dataloader
# test_loader = torch.utils.data.DataLoader(test_dset, 6, shuffle=True, num_workers=1, )



######################################################################
# 5.Initialize the GradCAM
######################################################################
print('\n' + '5.Initialize the GradCAM:')
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)


######################################################################
# 6.Select the input image category
######################################################################
print('\n' + '5.Initialize the GradCAM:')
targets = None


######################################################################
# 7.Calculate the CAM
######################################################################
print('\n' + '7.Calculate the CAM:')

'''3D resize function'''
'''
    input:numpy(H*W*C)
          newSzie:(C*H*W)
    output:numpy(H*W*C)
'''
def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing格式转换
    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

# for every patient
for i in tqdm(range(len(data))):
    # 1.prepare the data 1: heatmap
    grayscale_cam = cam(input_tensor=data[i].unsqueeze(0), targets=targets)
    grayscale_cam = grayscale_cam[0,:]   # 112*112*12  tensor
    # trans to itk
    grayscale_cam_itk = sitk.GetImageFromArray(np.array(grayscale_cam),)
    # 3D resize  (C*H*W)
    grayscale_cam_itk_resized = resize_image_itk(grayscale_cam_itk, (90, 112, 112), resamplemethod=sitk.sitkLinear)
    # trans to numpy(90,112,112)
    grayscale_cam_itk_resized_np = sitk.GetArrayFromImage(grayscale_cam_itk_resized)

    # 2.prepare the data 2: original image
    # original img (112,112,90)
    original_img = data_np[0].squeeze().astype('float32')/255

    # for every section of one patient
    for j in range(original_img.shape[0]):   # len(data_np)
        # cam_image = show_cam_on_image(original_img[i], grayscale_cam_itk_resized_np[:, :, i], use_rgb=True,image_weight=0.8)
        cam_image = show_cam_on_image(original_img[j], grayscale_cam_itk_resized_np[:, :, j], image_weight=0.8)
        # save the img
        cam_image = cv2.resize(cam_image, None, fx=2, fy=2)
        # # show the img
        # cv2.imshow('cam_image', cam_image)
        # cv2.waitKey(0)
        # write the img
        cam_img_path = name[i][:-1] + 'Heatmap_Conv'+str(conv_layer_num)
        # if path not exist, build
        if not os.path.exists(cam_img_path):
            os.mkdir(cam_img_path)
        cv2.imwrite(cam_img_path+'/'+str(j)+'.png', cam_image)



# # show the cam and original img on a picture  (112*112*90, 112*112*12--numpy)
# cam_image = show_cam_on_image(original_img[45], grayscale_cam_itk_resized_np[:,:,45], use_rgb=True)   # data[0].unsqueeze(0) !!
#
# # save the img
# cv2.imshow('cam_image',cam_image)
# cv2.waitKey(0)
# # cv2.imwrite(c, cam_image)




end = time.time()
print('Time:',end - start)








