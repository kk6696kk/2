from __future__ import print_function
import argparse
import sys
# sys.path.append('/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold')
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from network.Denseformer_no_Dialated import DenseNet
from lib.logger import Logger, savefig
from lib.misc import AverageMeter, mkdir_p
from lib import dataset_new_CT as dataset
from lib import utils_acc_auc as utils
import os
import numpy as np
from lib.Pretrain_freeze import load_pretrained_encoder_parameter_to_CTmodel


################# Parse arguments
parser = argparse.ArgumentParser(description='Training')

################# Datasets
parser.add_argument('-d', '--data', default='/data2/lwy/dataset/0_255', type=str)
parser.add_argument('-cuda', default='4', type=str, metavar='N',
                    help='the GPU (default: 0)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
################# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=6, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=6, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
################# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/data2/lwy/projevt/Lung/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/output_173_90_112_112/12_15_nopretrained_layer8_96_noDialated_Tran_Dialated_AC_in_DenseNet9_(CT2_second_hu_0_255)_(utils_acc_auc)_5fold_bs4_1', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
################# Transformer
parser.add_argument('--stand_dim', default=96, type=int, metavar='N',
                    help='the word size of the transformer')
parser.add_argument('--transformer_layers', default=8, type=int, metavar='N',
                    help='the layers of the transformer')
parser.add_argument('--transformer_heads', default=1, type=int, metavar='N',
                    help='the value can be divided by the stand_dim')
################# transfer learning
parser.add_argument('--transfer_learning', default=False, type=bool, metavar='N',
                    help='load the pretrained encoder parameter')

args = parser.parse_args()


################# set the cpu numbers
torch.set_num_threads(args.workers)

################# assign the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
use_cuda = torch.cuda.is_available()

################# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            # nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
                nn.init.constant_(w.bias, 0)
    if classname.find('Linear') != -1:
        if hasattr(w, 'weight'):
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('BatchNorm') != -1:
        if hasattr(w, 'weight') and w.weight is not None:
            nn.init.constant_(w.weight, 1)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)


def train_val(model, train_loader1, test_loader1, fold_index):
    best_acc = 0
    best_auc = 0

    state = {k: v for k, v in args._get_kwargs()}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # load the Encoder parameter
    if args.transfer_learning:
        path = '/data2/lwy/projevt/Lung/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/output/Encoder_Decoder_model/5_7_BS16_Encoder_Decoder/1/model_best.pth.tar'
        model = load_pretrained_encoder_parameter_to_CTmodel(model, path)
        print('Pretrained parameter load successful!')

    logger = Logger(os.path.join(args.checkpoint, fold_index, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train AUC.', 'Valid AUC.'])
    logger.append_str(['Epoch  ', 'Batchsize', 'LR        ', 'Dropout', 'gamma   ', 'momentum', 'weight-decay'])
    logger.append([args.epochs, args.train_batch, args.lr, args.drop, args.gamma, args.momentum, args.weight_decay])

    for epoch in range(0, args.epochs):
        utils.adjust_learning_rate(state, optimizer, epoch, args.schedule, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_auc = utils.train(train_loader1, model, criterion, optimizer, use_cuda, args.train_batch)
        test_loss, test_acc, test_auc, test_gt_list_array, test_pred_list_array, test_names = utils.test(test_loader1, model, criterion, use_cuda, args.train_batch)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_auc, test_auc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # best_auc = max(test_auc, best_auc)
        if test_auc > best_auc:
            best_auc = test_auc
            np.save(os.path.join(args.checkpoint, fold_index, 'test_names.npy'), test_names)
            np.save(os.path.join(args.checkpoint, fold_index, 'raw_auc_curve_gt_all.npy'), test_gt_list_array)
            np.save(os.path.join(args.checkpoint, fold_index, 'raw_auc_curve_prob_all.npy'), test_pred_list_array)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=os.path.join(args.checkpoint, fold_index))

    logger.append_str(['best_acc:', str(best_acc), 'best_auc:', str(best_auc)])
    logger.close()
    logger.plot()
    savefig(os.path.join(os.path.join(args.checkpoint, fold_index), 'log.eps'))

    return best_acc, best_auc


def initialize_datasets():
    if not os.path.isdir(args.checkpoint):
        mkdir_p(os.path.join(args.checkpoint, '1'))
        mkdir_p(os.path.join(args.checkpoint, '2'))
        mkdir_p(os.path.join(args.checkpoint, '3'))
        mkdir_p(os.path.join(args.checkpoint, '4'))
        mkdir_p(os.path.join(args.checkpoint, '5'))

        ###########    dataloader
    alldir = os.path.join(args.data)
    print(alldir)
    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, \
    train_loader4, test_loader4, train_loader5, test_loader5 = dataset.loaderloader(alldir, args.train_batch)
    print("finish loading dataset")

    return train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, \
    train_loader4, test_loader4, train_loader5, test_loader5


def main1(train_loader1, test_loader1):
    ###########    model
    model1 = DenseNet(
        stand_dim=args.stand_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout_p=0
    )
    model1.cuda()

    ############ I dont know it
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model1.parameters()) / 1000000.0))

    ############### Train
    best_acc1, best_auc1 = train_val(model1, train_loader1, test_loader1, '1')
    print('Best acc1:', best_acc1)

    return best_acc1, best_auc1


def main2(train_loader2, test_loader2):
    ###########    model
    model2 = DenseNet(
        stand_dim=args.stand_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout_p=0
    )
    model2.cuda()

    ############ I dont know it
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model2.parameters()) / 1000000.0))

    ############### Train
    best_acc2, best_auc2 = train_val(model2, train_loader2, test_loader2, '2')
    print('Best acc2:', best_acc2)

    return best_acc2, best_auc2


def main3(train_loader3, test_loader3):
    ###########    model
    model3 = DenseNet(
        stand_dim=args.stand_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout_p=0
    )
    model3.cuda()

    ############ I dont know it
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model3.parameters()) / 1000000.0))

    ############### Train
    best_acc3, best_auc3 = train_val(model3, train_loader3, test_loader3, '3')
    print('Best acc13:', best_acc3)

    return best_acc3, best_auc3


def main4(train_loader4, test_loader4):
    ###########    model
    model4 = DenseNet(
        stand_dim=args.stand_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout_p=0
    )
    model4.cuda()

    ############ I dont know it
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model4.parameters()) / 1000000.0))

    ############### Train
    best_acc4, best_auc4 = train_val(model4, train_loader4, test_loader4, '4')
    print('Best acc4:', best_acc4)

    return best_acc4, best_auc4


def main5(train_loader5, test_loader5):
    ###########    model
    model5 = DenseNet(
        stand_dim=args.stand_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout_p=0
    )
    model5.cuda()

    ############ I dont know it
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model5.parameters()) / 1000000.0))

    ############### Train
    best_acc5, best_auc5 = train_val(model5, train_loader5, test_loader5, '5')
    print('Best acc5:', best_acc5)

    return best_acc5, best_auc5


def eval_mean_acc(best_acc1, best_acc2, best_acc3, best_acc4, best_acc5, best_auc1, best_auc2, best_auc3, best_auc4, best_auc5):
    mean_best_acc = (best_acc1 + best_acc2 + best_acc3 + best_acc4 + best_acc5) / 5
    mean_best_auc = (best_auc1 + best_auc2 + best_auc3 + best_auc4 + best_auc5) / 5

    print('Best acc:', best_acc1, best_acc2, best_acc3, best_acc4, best_acc5)
    print('mean_best_acc:', mean_best_acc)

    print('Best auc:', best_auc1, best_auc2, best_auc3, best_auc4, best_auc5)
    print('mean_best_auc:', mean_best_auc)

    with open(os.path.join(args.checkpoint, 'res.txt'), 'w') as f:
        f.writelines(['best_acc1   ', 'best_acc2   ', 'best_acc3   ', 'best_acc4   ', 'best_acc5   '])
        f.writelines('\n')
        f.writelines(str([best_acc1, best_acc2, best_acc3, best_acc4, best_acc5]))
        f.writelines('\n')
        f.writelines('mean_best_acc')
        f.writelines('\n')
        f.writelines(str(mean_best_acc))
        f.writelines('\n')

        f.writelines('\n')
        f.writelines(['best_auc1   ', 'best_auc2   ', 'best_auc3   ', 'best_auc4   ', 'best_auc5   '])
        f.writelines('\n')
        f.writelines(str([best_auc1, best_auc2, best_auc3, best_auc4, best_auc5]))
        f.writelines('\n')
        f.writelines('mean_best_auc')
        f.writelines('\n')
        f.writelines(str(mean_best_auc))


if __name__ == '__main__':
    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, \
    train_loader4, test_loader4, train_loader5, test_loader5 = initialize_datasets()

    best_acc1, best_auc1 = main1(train_loader1, test_loader1)
    best_acc2, best_auc2 = main2(train_loader2, test_loader2)
    best_acc3, best_auc3 = main3(train_loader3, test_loader3)
    best_acc4, best_auc4 = main4(train_loader4, test_loader4)
    best_acc5, best_auc5 = main5(train_loader5, test_loader5)
    eval_mean_acc(best_acc1, best_acc2, best_acc3, best_acc4, best_acc5, best_auc1, best_auc2, best_auc3, best_auc4, best_auc5)
