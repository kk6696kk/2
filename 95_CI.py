import numpy as np
from scipy import stats

# 假设你有一组样本数据，存储在一个NumPy数组中
acc = \
[84.21052631578948, 56.75675675675676, 67.56756756756756, 59.45945945945946, 72.97297297297297]

auc = \
[0.6543778801843319, 0.5148809523809524, 0.6466666666666667, 0.5238095238095237, 0.4592592592592593]


# 使用列表推导式将每个元素除以 100
acc = [x / 100 for x in acc]

# acc
data_acc = np.array(acc)

# 计算平均值和标准误差
mean_acc = np.mean(data_acc)
std_err_auc = stats.sem(data_acc)

# 计算置信区间c
confidence_interval = stats.t.interval(0.95, len(data_acc)-1, loc=mean_acc, scale=std_err_auc)
# 对元组中的每个元素进行四舍五入
confidence_interval = tuple(round(x, 3) for x in confidence_interval)


print("acc_95%置信区间:", confidence_interval)


# auc
data_auc = np.array(auc)

# 计算平均值和标准误差
mean_auc = np.mean(data_auc)
std_err_auc = stats.sem(data_auc)

# 计算置信区间c
confidence_interval = stats.t.interval(0.95, len(data_auc)-1, loc=mean_auc, scale=std_err_auc)
# 对元组中的每个元素进行四舍五入
confidence_interval = tuple(round(x, 3) for x in confidence_interval)

print("auc_95%置信区间:", confidence_interval)


print("acc_avg: ", sum(acc)/5)
print("auc_avg: ", sum(auc)/5)