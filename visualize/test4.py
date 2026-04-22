# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# #定义坐标轴
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# # z = np.linspace(0,13,1000)
# # x = 5*np.sin(z)
# # y = 5*np.cos(z)
# # zd = 13*np.random.random(100)
# # xd = 5*np.sin(zd)
# # yd = 5*np.cos(zd)
# arr = [[[1.0, 0.8, 0.7], [1.0, 0.6, 0.3]], [[1.0, 0.8, 0.7], [1.0, 0.6, 0.3]]]
# print(arr.size())
# ax1.scatter3D(xd,yd,zd, cmap='b')  #绘制散点图
# print(xd)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0, 11, 0.5)
print(x)
y = np.arange(0, 11, 0.5)
z = np.arange(0, 11, 0.5)

x, y, z = np.meshgrid(x, y, z)

color = np.array([i+j+k for i, j, k in zip(x, y, z)])
print(color.shape)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c=color)
# plt.savefig('/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/visualize/res/112_112_112/test3/fig.png')
plt.show()

# def show_3d_hot_map(array, save_dir):
#     x = np.arange(0, 10, 0.05)
#     y = np.arange(0, 10, 0.05)
#     z = np.arange(0, 10, 0.05)
#     x, y, z = np.meshgrid(x, y, z)
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x, y, z, c=array)
#     plt.savefig(save_dir)

