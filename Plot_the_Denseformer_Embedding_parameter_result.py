from matplotlib import pyplot as plt

# Embedding coordinate
x_label = ['0', '32', '64', '96', '128', '256', '512']
x = range(len(x_label))
# 90*112*112
# y_auc = [0.7950, 0.8258, 0.7964, 0.8076, 0.8065, 0.7680, 0.7286]
# y_acc = [0.7739, 0.7950, 0.7778, 0.8017, 0.7978, 0.7672, 0.7333]

# 112*112*112
y_auc = [0.7950, 0.8258, 0.7964, 0.8069, 0.8065, 0.7680, 0.7286]
y_acc = [0.7739, 0.7950, 0.7778, 0.8007, 0.7978, 0.7672, 0.7333]


# plot
plt.plot(x, y_auc, color='sandybrown', marker='o')
plt.plot(x, y_acc, color='dodgerblue', marker='o')
# plot the dimension
plt.xticks(range(7), labels=x_label)
# plot the value
for x_, y_auc_, y_acc_ in zip(x, y_auc, y_acc):
    if x_ != 5:
        plt.text(x_, y_auc_, y_auc_, ha='center', va='bottom', fontsize=10)
        plt.text(x_, y_acc_, y_acc_, ha='center', va='bottom', fontsize=10)

plt.text(x[5], y_auc[5], y_auc[5], ha='center', va='bottom', fontsize=10)
plt.text(x[5], y_acc[5], y_acc[5], ha='center', va='top', fontsize=10)
plt.xlabel('Embedding Dimension')
plt.ylabel('Evaluation Metrics')
plt.legend(['AUC','ACC'])

plt.savefig('Parameter_Embedding.tiff')
plt.show()


# from matplotlib import pyplot as plt
#
# # Embedding coordinate
# x_label = ['0', '32', '64', '96', '128', '256', '512']
# x = range(len(x_label))
# y_auc = [0.7950, 0.8258, 0.7964, 0.8076, 0.8065, 0.7680, 0.7286]
# y_acc = [0.7739, 0.7950, 0.7778, 0.8017, 0.7978, 0.7672, 0.7333]
#
#
# # plot
# plt.plot(x, y_auc, color='sandybrown', marker='o')
# plt.plot(x, y_acc, color='dodgerblue', marker='o')
# # plot the dimension
# plt.xticks(range(7), labels=x_label)
# # plot the value
# for x_, y_auc_, y_acc_ in zip(x, y_auc, y_acc):
#     if x_ != 5:
#         plt.text(x_, y_auc_, y_auc_, ha='center', va='bottom', fontsize=10)
#         plt.text(x_, y_acc_, y_acc_, ha='center', va='bottom', fontsize=10)
#
# plt.text(x[5], y_auc[5], y_auc[5], ha='center', va='bottom', fontsize=10)
# plt.text(x[5], y_acc[5], y_acc[5], ha='center', va='top', fontsize=10)
# plt.xlabel('Embedding Dimension')
# plt.ylabel('Evaluation Metrics')
# plt.legend(['AUC','ACC'])
#
# plt.savefig('Parameter_Embedding.png')
# plt.show()
#
#
# from matplotlib import pyplot as plt
#
# # Embedding coordinate
# x_label = ['0', '32', '64', '96', '128', '256', '512']
# x = range(len(x_label))
# y_auc = [0.7950, 0.8258, 0.7964, 0.8076, 0.8065, 0.7680, 0.7286]
# y_acc = [0.7739, 0.7950, 0.7778, 0.8017, 0.7978, 0.7672, 0.7333]
#
#
# # plot
# plt.plot(x, y_auc, color='sandybrown', marker='o')
# plt.plot(x, y_acc, color='dodgerblue', marker='o')
# # plot the dimension
# plt.xticks(range(7), labels=x_label)
# # plot the value
# for x_, y_auc_, y_acc_ in zip(x, y_auc, y_acc):
#     if x_ != 5:
#         plt.text(x_, y_auc_, y_auc_, ha='center', va='bottom', fontsize=10)
#         plt.text(x_, y_acc_, y_acc_, ha='center', va='bottom', fontsize=10)
#
# plt.text(x[5], y_auc[5], y_auc[5], ha='center', va='bottom', fontsize=10)
# plt.text(x[5], y_acc[5], y_acc[5], ha='center', va='top', fontsize=10)
# plt.xlabel('Embedding Dimension')
# plt.ylabel('Evaluation Metrics')
# plt.legend(['AUC','ACC'])
#
# plt.savefig('Parameter_Embedding.png')
# plt.show()
#
#
# from matplotlib import pyplot as plt
#
# # Embedding coordinate
# x_label = ['0', '32', '64', '96', '128', '256', '512']
# x = range(len(x_label))
# y_auc = [0.7950, 0.8258, 0.7964, 0.8076, 0.8065, 0.7680, 0.7286]
# y_acc = [0.7739, 0.7950, 0.7778, 0.8017, 0.7978, 0.7672, 0.7333]
#
#
# # plot
# plt.plot(x, y_auc, color='sandybrown', marker='o')
# plt.plot(x, y_acc, color='dodgerblue', marker='o')
# # plot the dimension
# plt.xticks(range(7), labels=x_label)
# # plot the value
# for x_, y_auc_, y_acc_ in zip(x, y_auc, y_acc):
#     if x_ != 5:
#         plt.text(x_, y_auc_, y_auc_, ha='center', va='bottom', fontsize=10)
#         plt.text(x_, y_acc_, y_acc_, ha='center', va='bottom', fontsize=10)
#
# plt.text(x[5], y_auc[5], y_auc[5], ha='center', va='bottom', fontsize=10)
# plt.text(x[5], y_acc[5], y_acc[5], ha='center', va='top', fontsize=10)
# plt.xlabel('Embedding Dimension')
# plt.ylabel('Evaluation Metrics')
# plt.legend(['AUC','ACC'])
#
# plt.savefig('Parameter_Embedding.png')
# plt.show()


