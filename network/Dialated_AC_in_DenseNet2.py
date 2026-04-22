import torch
import torch.nn as nn
import torch.nn.functional as F
# pass


class My_AC_layer1(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(My_AC_layer1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (5, 5, 5), stride=1, padding=2, bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (1, 5, 5), stride=1, padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (5, 1, 5), stride=1, padding=(2, 0, 2), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (5, 5, 1), stride=1, padding=(2, 2, 0), bias=False),
            nn.BatchNorm3d(outchannels))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4


class Dialated_AC_layer_in_Dense(nn.Module):
    def __init__(self, Stage_i, inchannels, outchannels):
        super(Dialated_AC_layer_in_Dense, self).__init__()

        if Stage_i == 1:
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
        elif Stage_i == 2:
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
        elif Stage_i == 3:
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
        elif Stage_i == 4 or Stage_i == 5:
            self.conv1 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm3d(outchannels))
            self.conv2 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(outchannels))
            self.conv3 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 1, 3), stride=1, padding=(1, 0, 1), bias=False),
                nn.BatchNorm3d(outchannels))
            self.conv4 = nn.Sequential(
                nn.Conv3d(inchannels, outchannels, (3, 3, 1), stride=1, padding=(1, 1, 0), bias=False),
                nn.BatchNorm3d(outchannels))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4


class SE_block(nn.Module):
    def __init__(self, inchannels, reduction=16):
        super(SE_block, self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.FC1 = nn.Linear(inchannels, inchannels//reduction)
        self.FC2 = nn.Linear(inchannels//reduction, inchannels)

    def forward(self, x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return model_input * x


class dense_layer(nn.Module):
    def __init__(self, Stage_i, inchannels, outchannels):
        super(dense_layer, self).__init__()
        self.block = nn.Sequential(
            Dialated_AC_layer_in_Dense(Stage_i, inchannels, outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),

            # Dialated_AC_layer_in_Dense(Stage_i, outchannels, outchannels),
            # nn.BatchNorm3d(outchannels),
            # nn.ELU(),

            SE_block(outchannels),
            nn.MaxPool3d(2, 2),
        )

    def forward(self, x):
        new_features = self.block(x)
        x = F.max_pool3d(x, 2)
        # print('new_feature_size', new_features.size())
        # print('x', x.size())
        x = torch.cat([x, new_features], 1)

        return x


class DenseNet(nn.Module):
    def __init__(self, nb_filter=8, nb_block=4):
        super(DenseNet, self).__init__()

        self.nb_block = nb_block

        # self.pre = My_AC_layer1(1, nb_filter)
        self.pre = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1, dilation=2),
            nn.ELU(),
        )

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
            outchannels = inchannels * 2
            if i+1 <= 3:
                blocks.append(dense_layer(i + 1, inchannels, outchannels))
            else:
                blocks.append(dense_layer(i + 1, inchannels, outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):
        x = self.pre(x)

        x = self.block(x)

        x = self.gap(x)

        x = torch.reshape(x, (x.size(0), -1))

        x = self.FC_layers(x)

        return x


if __name__=='__main__':
    model = DenseNet(8, 5)

    datas = []
    data1 = torch.rand(1, 1, 80, 112, 112)
    # data1 = torch.rand(1, 1, 91, 109, 91)
    datas.append(data1)
    for data in datas:
        output = model(data)
        print(output.size())

        break
