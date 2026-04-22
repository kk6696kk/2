import torch
import torch.nn as nn
import torch.nn.functional as F


class dense_layer(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(dense_layer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2, 2),
        )

    def forward(self, x):
        new_features = self.block(x)
        x = F.max_pool3d(x, 2)
        x = torch.cat([x, new_features], 1)

        return x


class DenseNet(nn.Module):
    def __init__(self, nb_filter=8, nb_block=5, use_gender=True):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(DenseNet, self).__init__()

        self.nb_block = nb_block

        self.use_gender = use_gender

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
                                       )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer(inchannels, outchannels))
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
    model = DenseNet(8, 5, False)

    datas = []
    data1 = torch.rand(4, 1, 80, 112, 112)
    datas.append(data1)

    for data in datas:
        output = model(data)
        print(output.size())
        break
