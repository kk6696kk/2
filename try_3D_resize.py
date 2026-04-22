import numpy as np
import SimpleITK as sitk
from glob import glob

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)   # spacing格式转换
    resampler.SetReferenceImage(itkimage)   # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


# image_path = 'F:\\image'    # 文件夹路径
# image_file = glob(image_path +"/*")   # 遍历该文件夹下所有的文件
# for i in range(len(image_file)):
#     itkimage = sitk.ReadImage(image_file[i])    # 遍历文件夹下每一张图片
#     itkimgResampled = resize_image_itk(itkimage, (128,128,128),
#                                    resamplemethod= sitk.sitkNearestNeighbor)
#                                     # 目标size为(128,128,128)
#                                     # 这里要注意：mask用最近邻插值，CT图像用线性插值
#     sitk.WriteImage(itkimgResampled,'F:\\image' + image_file[i][len(image_path):])


if __name__ == '__main__':
    itkimage = np.ones([90,128,128])
    itkimage = sitk.GetImageFromArray(itkimage,)
    itkimgResampled = resize_image_itk(itkimage, (128, 128, 128),
                                       resamplemethod=sitk.sitkLinear)
    npImage = sitk.GetArrayFromImage(itkimgResampled)
    print('end!')
