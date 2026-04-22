import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# Reproduced D3Net of CVPR2021


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)






class AC_layer(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(AC_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
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
            nn.Conv3d(inchannels, inchannels, (3, 3, 3), stride=1, padding=1, bias=False, dilation=1),
            nn.BatchNorm3d(inchannels),)

        self.Net2 = nn.Sequential(
            nn.Conv3d(inchannels*2, inchannels, (3, 3, 3), stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm3d(inchannels),)
        self.Net2_1 = nn.Sequential(
            nn.Conv3d(inchannels * 2, inchannels, (3, 3, 3), stride=1, padding=1, bias=False, dilation=1),
            nn.BatchNorm3d(inchannels), )

        self.Net3 = nn.Sequential(
            nn.Conv3d(inchannels*3, inchannels, (3, 3, 3), stride=1, padding=4, bias=False, dilation=4),
            nn.BatchNorm3d(inchannels),)
        self.Net3_1 = nn.Sequential(
            nn.Conv3d(inchannels * 3, inchannels, (3, 3, 3), stride=1, padding=1, bias=False, dilation=1),
            nn.BatchNorm3d(inchannels), )
        self.Net3_2 = nn.Sequential(
            nn.Conv3d(inchannels * 3, inchannels, (3, 3, 3), stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm3d(inchannels), )

    def forward(self, x):
        out1 = self.Net1(x)

        x_and_out1 = torch.cat([x, out1], 1)
        out2 = self.Net2(x_and_out1) + self.Net2_1(x_and_out1)

        x_and_out_1_and_out2 = torch.cat([x_and_out1, out2], 1)
        out3 = self.Net3(x_and_out_1_and_out2) + self.Net3_1(x_and_out_1_and_out2) + self.Net3_2(x_and_out_1_and_out2)

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

            AC_layer(inchannels, inchannels),
            nn.BatchNorm3d(inchannels),
            nn.ELU(),

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


class dense_layer_final(nn.Module):
    def __init__(self, inchannels):
        super(dense_layer_final, self).__init__()
        self.block = nn.Sequential(

            AC_layer(inchannels, inchannels),
            nn.BatchNorm3d(inchannels),
            nn.ELU(),

            nn.MaxPool3d(2, 2)
        )

    def forward(self, x):
        new_features = self.block(x)

        x = F.max_pool3d(x, 2)

        out = torch.cat([x, new_features], 1)

        return out


class DenseNet(nn.Module):
    def __init__(self, nb_filter=8, nb_block=4,
                 num_classes=2,
                 stand_dim=128,
                 transformer_layers=1,
                 transformer_heads=1,
                 dropout_p=0
                 ):
        super(DenseNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1, dilation=2),
            nn.ELU(),)

        self.block, last_channels = self._make_block(nb_filter, nb_block)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.FC_layers = nn.Sequential(nn.Linear(last_channels, 512, bias=True),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(512, 32, bias=True),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(32, 2, bias=True),

                                       # nn.Softmax(dim=1)
                                       )

        # parameters
        self.stand_dim = stand_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.dropout_p = dropout_p

        # networks
        # self.to_stand_feature_0 = nn.Linear(8*80*102*102, self.stand_dim)
        self.to_stand_feature_1 = nn.Linear(24*20*26*26, self.stand_dim)
        self.to_stand_feature_2 = nn.Linear(72*10*13*13, self.stand_dim)
        self.to_stand_feature_3 = nn.Linear(216*5*6*6, self.stand_dim)
        self.to_stand_feature_4 = nn.Linear(648, self.stand_dim)
        self.to_stand_feature_5 = nn.Linear(num_classes, self.stand_dim)
        self.FC = nn.Linear(5 * stand_dim, num_classes)  # 4:layers+1
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.AVGpool = nn.AdaptiveAvgPool2d((5,5))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # fusion block
        self.transformer = Transformer(
            width=self.stand_dim,
            layers=self.transformer_layers,
            heads=self.transformer_heads
        )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            if i < 4:
                blocks.append(dense_layer(inchannels))
                outchannels = inchannels * 2
                inchannels = inchannels + outchannels
            else:
                blocks.append(dense_layer_final(inchannels))
                outchannels = inchannels
                inchannels = inchannels + outchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):  # [4, 1, 80, 112, 112]
        x = self.pre(x)  # [4, 8, 80, 102, 102]
        # feature_0 = x
        feature_1 = None
        feature_2 = None
        feature_3 = None
        feature_4 = None
        feature_5 = None
        # x2 = self.block(x1)
        for name,block in self.block.named_children():
            x = block(x)
            if name == '0':
                feature_1 = x
            elif name == '1':
                feature_2 = x
            elif name == '2':
                feature_3 = x

        '''
        name: 0
        x: torch.Size([4, 24, 40, 51, 51])
        name: 1
        x: torch.Size([4, 72, 20, 25, 25])
        name: 2
        x: torch.Size([4, 216, 10, 12, 12])
        name: 3
        x: torch.Size([4, 648, 5, 6, 6])
        '''

        self.feature = x

        x3 = self.gap(x)  # torch.Size([4, 648, 1, 1, 1])
        x4 = torch.reshape(x3, (x3.size(0), -1))  # torch.Size([4, 648])
        feature_4 = x4

        x5 = self.FC_layers(x4)
        feature_5 = x5

        # flatten the feature to B*(C*H*W)  self.AVGpool
        # feature_0 = torch.flatten(feature_0, 1)              # [4, 8, 80, 102, 102]
        feature_1 = torch.flatten(self.maxpool(feature_1), 1)                # [4, 24, 40, 51, 51]
        feature_2 = torch.flatten(self.maxpool(feature_2), 1)                # [4, 72, 20, 25, 25]
        feature_3 = torch.flatten(self.maxpool(feature_3), 1)                # [4, 216, 10, 12, 12]
        feature_4 = torch.flatten(feature_4, 1)                # [4, 648]
        feature_5 = torch.flatten(feature_5, 1)                # [4, 2]

        # feature_vector to stand size B*(3*32*32)
        # stand_feature_0 = self.sigmoid(self.to_stand_feature_0(feature_0))
        stand_feature_1 = self.sigmoid(self.to_stand_feature_1(feature_1))
        stand_feature_2 = self.sigmoid(self.to_stand_feature_2(feature_2))
        stand_feature_3 = self.sigmoid(self.to_stand_feature_3(feature_3))
        stand_feature_4 = self.sigmoid(self.to_stand_feature_4(feature_4))
        stand_feature_5 = self.sigmoid(self.to_stand_feature_5(feature_5))

        # cat the features and reshape the size: (batch,layers,stand_dim)
        stand_feature_cat = torch.cat((stand_feature_1,
                                       stand_feature_2,
                                       stand_feature_3,
                                       stand_feature_4,
                                       stand_feature_5
                                       ), 1)  # seq_length:句子的长度   d_model:是每一个单词本来的词向量长度
        stand_feature_cat = stand_feature_cat.reshape(-1, 5,
                                                      self.stand_dim)  # (batch,layers+1,stand_dim) = (batch_size, seq_length, d_model)

        # feature fusion
        features_after_transformer = self.transformer(stand_feature_cat)

        # reshape the features size:(batch,(layers+1)*stand_dim)
        features = torch.flatten(features_after_transformer, 1)

        # add the relu
        features = self.relu(features)

        # output
        output = self.FC(features)

        return output


if __name__=='__main__':
    model = DenseNet(8, 4)
    print(model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    datas = []
    data1 = torch.rand(4, 1, 90, 112, 112)
    datas.append(data1)
    for data in datas:
        print(data.size())
        output = model(data)
        print(output.size())

        break
