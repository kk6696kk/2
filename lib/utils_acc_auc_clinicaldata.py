import sys

sys.path.append("/data2/lwy/projevt/Lung/VGG_egfr8_Dialated_AC_in_Dense_K_fold_new")
from lib.logger import Logger, savefig
from lib.misc import AverageMeter, mkdir_p
from lib.eval import accuracy
import time
import torch
import torch.nn as nn
from progress.bar import Bar as Bar
import os
import shutil
from torch.autograd import Variable
from network.MLP import Linear, Transformer, MLP_Transformer_enconder
import lib.dataset_clinical_data as dataset
import torch.optim as optim
import cv2
from torchvision import transforms
from lib.eval_acc_auc import AUC_score
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '9'


def adjust_learning_rate(state, optimizer, epoch, schedule, gamma):
    # global state
    if epoch in schedule:
        state['lr'] *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))

    pred_score_all, gt_all = [], []

    for batch_idx, (inputs, targets, names) in enumerate(train_loader):  # data dim is 19
        optimizer.zero_grad()
        if use_cuda:
            targets = targets.cuda()
        targets = torch.autograd.Variable(targets)
        data_time.update(time.time() - end)

        # output
        if use_cuda:
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        outputs = model(inputs)

        # loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Acc
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))

        # print('input:')
        # print('output:', outputs)
        # print('target', target)
        # print('Acc', prec1)
        # Auc
        outputs_numpy = outputs.data.cpu().numpy()
        # print('outputs_numpy:',outputs_numpy)
        targets_numpy = targets.data.cpu().numpy()
        for i in range(len(targets_numpy)):  # probability to classes
            outputs_numpy_0_0r_1 = 1 if outputs_numpy[i][1] > outputs_numpy[i][0] else 0
            # print('AUC', outputs_numpy[i], outputs_numpy[i][1], outputs_numpy_0_0r_1)
            pred_score_all.append(outputs_numpy_0_0r_1)
            gt_all.append(targets_numpy[i])

        # add
        losses.update(loss.item(), 1)
        top1.update(prec1.item(), 1)
        top5.update(prec5.item(), 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()

    # AUC
    auc = AUC_score(np.array(gt_all), np.array(pred_score_all))

    return (losses.avg, top1.avg, auc)


def test(val_loader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    pred_score_all, gt_all = [], []

    for batch_idx, (inputs, targets, names) in enumerate(val_loader):
        # for batch_idx, (input, targets) in enumerate(val_loader):
        # print(names)
        if use_cuda:
            targets = targets.cuda()
        targets = torch.autograd.Variable(targets)
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
        # print('prec1:', prec1)

        # Auc
        outputs_numpy = outputs.data.cpu().numpy()
        targets_numpy = targets.data.cpu().numpy()
        for i in range(len(targets_numpy)):
            outputs_numpy_0_0r_1 = 1 if outputs_numpy[i][1] > outputs_numpy[i][0] else 0
            pred_score_all.append(outputs_numpy_0_0r_1)
            gt_all.append(targets_numpy[i])

        losses.update(loss.item(), 1)
        top1.update(prec1.item(), 1)
        top5.update(prec5.item(), 1)

        # print('top1:', top1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()

    # AUC
    # print(gt_all)
    # print(pred_score_all)
    auc = AUC_score(np.array(gt_all), np.array(pred_score_all))
    # print(auc)

    return (losses.avg, top1.avg, auc)


if __name__ == '__main__':
    # model = Linear(19, 15 , 10 , 2, dropout_p=0)
    model = MLP_Transformer_enconder(
        in_dim=19,
        n_hidden_1=76,
        n_hidden_2=38,
        out_dim=2,
        stand_dim=72,
        transformer_layers=6,
        transformer_heads=8,
        dropout_p=0
    )
    # model = Transformer(
    #     width=19,
    #     layers=6,
    #     heads=1
    #     # attn_mask=self.build_attention_mask()
    # )
    # model.cuda()
    alldir = '/data2/lwy/dataset/clinical_data.xlsx'

    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, train_loader4, test_loader4, train_loader5, test_loader5 = dataset.loaderloader(
        alldir, 6, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    for epoch in range(0, 5):
        loss, acc, auc = train(train_loader1, model, criterion, optimizer, False)
        print('\n',loss, acc, auc)
        loss_, acc_, auc_ = test(test_loader1, model, criterion, False)
        print(loss_, acc_, auc_)

