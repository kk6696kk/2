import torch
import torch.nn as nn
import torch.nn.functional as F
# Reproduced D3Net of CVPR2021
# dialated AC block in dialated DeNet


class Dialated_AC_layer_in_Dense(nn.Module):
    def __init__(self, d, inchannels, outchannels):
        super(Dialated_AC_layer_in_Dense, self).__init__()

        if d == 1:
            self.conv1 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=(1, 1, 1), bias=False, dilation=1),
                nn.BatchNorm3d(outchannels))
            self.conv2 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, dilation=1),
                nn.BatchNorm3d(outchannels))
            self.conv3 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 1, 3), stride=1, padding=(1, 0, 1), bias=False, dilation=1),
                nn.BatchNorm3d(outchannels))
            self.conv4 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 1), stride=1, padding=(1, 1, 0), bias=False, dilation=1),
                nn.BatchNorm3d(outchannels))
        elif d == 2:
            self.conv1 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=(2, 2, 2), bias=False, dilation=2),
                nn.BatchNorm3d(outchannels))
            self.conv2 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (1, 3, 3), stride=1, padding=(0, 2, 2), bias=False, dilation=2),
                nn.BatchNorm3d(outchannels))
            self.conv3 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 1, 3), stride=1, padding=(2, 0, 2), bias=False, dilation=2),
                nn.BatchNorm3d(outchannels))
            self.conv4 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 1), stride=1, padding=(2, 2, 0), bias=False, dilation=2),
                nn.BatchNorm3d(outchannels))
        elif d == 4:
            self.conv1 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=(4, 4, 4), bias=False, dilation=4),
                nn.BatchNorm3d(outchannels))
            self.conv2 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (1, 3, 3), stride=1, padding=(0, 4, 4), bias=False, dilation=4),
                nn.BatchNorm3d(outchannels))
            self.conv3 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 1, 3), stride=1, padding=(4, 0, 4), bias=False, dilation=4),
                nn.BatchNorm3d(outchannels))
            self.conv4 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 1), stride=1, padding=(4, 4, 0), bias=False, dilation=4),
                nn.BatchNorm3d(outchannels))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4


class D2Net(nn.Module):
    # outchannels == 4 * inchannels
    def __init__(self, inchannels):
        super(D2Net, self).__init__()
        self.Net1 = nn.Sequential(
            Dialated_AC_layer_in_Dense(1, inchannels, inchannels),
            nn.BatchNorm3d(inchannels),
            nn.ELU())

        self.Net2 = nn.Sequential(
            Dialated_AC_layer_in_Dense(2, inchannels*2, inchannels),
            nn.BatchNorm3d(inchannels),
            nn.ELU())

        self.Net3 = nn.Sequential(
            Dialated_AC_layer_in_Dense(4, inchannels*3, inchannels),
            nn.BatchNorm3d(inchannels),
            nn.ELU())

    def forward(self, x):
        out1 = self.Net1(x)

        x_and_out1 = torch.cat([x, out1], 1)
        out2 = self.Net2(x_and_out1)

        x_and_out_1_and_out2 = torch.cat([x_and_out1, out2], 1)
        out3 = self.Net3(x_and_out_1_and_out2)

        out = torch.cat([x_and_out_1_and_out2, out3], 1)
        return out


class Transition(nn.Module):
    def __init__(self, inchannels):
        # outchannels = inchannels // 2
        super(Transition, self).__init__()
        outchannels = inchannels // 2
        self.Net = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=1, bias=False, dilation=1),
            nn.MaxPool3d(2, 2))

    def forward(self, x):
        out = self.Net(x)

        return out


class dense_layer(nn.Module):
    def __init__(self, inchannels):
        super(dense_layer, self).__init__()
        self.block = nn.Sequential(

            D2Net(inchannels),
            nn.BatchNorm3d(inchannels * 4),
            nn.ELU(),

            Transition(inchannels * 4)
        )

    def forward(self, x):
        new_features = self.block(x)

        x = F.max_pool3d(x, 2)

        out = torch.cat([x, new_features], 1)

        return out


class DenseNet(nn.Module):
    def __init__(self, nb_filter=8, nb_block=5):
        super(DenseNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1, dilation=2),
            nn.ELU(),)

        self.block, last_channels = self._make_block(nb_filter, nb_block)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.FC_layers = nn.Sequential(nn.Linear(last_channels, 1024, bias=True),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(1024, 32, bias=True),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(32, 2, bias=True),

                                       nn.Softmax(dim=1)
                                       )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            blocks.append(dense_layer(inchannels))
            outchannels = inchannels * 2
            inchannels = inchannels + outchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):  # [1, 1, 80, 112, 112]
        x = self.pre(x)  # [1, 8, 70, 102, 102]

        x = self.block(x)  # [1, 1944, 2, 3, 3]

        x = self.gap(x)  # [1, 1944, 1, 1, 1]
        x = torch.reshape(x, (x.size(0), -1))  # [1, 1944]

        x = self.FC_layers(x)  # [1, 2]

        return x


if __name__=='__main__':
    model = DenseNet(8, 5)

    datas = []
    data1 = torch.rand(1, 1, 80, 112, 112)
    datas.append(data1)
    for data in datas:
        print(data.size())
        output = model(data)
        print(output.size())

        break
