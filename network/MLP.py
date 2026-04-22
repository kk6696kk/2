import torch.nn as nn
from collections import OrderedDict
import torch

import torch.nn.functional as F

def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)

class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class Swish(torch.nn.Module):
    """Construct an Swish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)

class Linear(nn.Module):
    def __init__(self, in_dim,
                 n_hidden_1, n_hidden_2,
                 out_dim, dropout_p):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        # print('layer1_output:',x.shape)   # layer1_output: torch.Size([6, 15])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        # print('layer2_output:', x.shape)   # layer2_output: torch.Size([6, 10])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        # print('layer3_output:', x.shape)   # layer3_output: torch.Size([6, 2])
        x = self.softmax(x)
        return x

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

class MLP_Transformer_enconder(nn.Module):
    def __init__(self, in_dim,
                 n_hidden_1,
                 n_hidden_2,
                 out_dim,
                 stand_dim=72,
                 transformer_layers=6,
                 transformer_heads=8,
                 dropout_p=0):
        super().__init__()
        # parameters
        self.stand_dim = stand_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.dropout_p = dropout_p

        # networks
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.to_stand_feature_0 = nn.Linear(in_dim, stand_dim)
        self.to_stand_feature_1 = nn.Linear(n_hidden_1, stand_dim)
        self.to_stand_feature_2 = nn.Linear(n_hidden_2, stand_dim)
        self.to_stand_feature_3 = nn.Linear(out_dim, stand_dim)
        self.FC = nn.Linear(4*stand_dim, out_dim)   # 4:layers+1  4*stand_dim
        # self.FC = nn.Linear(stand_dim, out_dim)  # 4:layers+1  4*stand_dim
        self.relu = nn.ReLU()
        self.swish = Swish()
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.LN0 = LayerNorm(19)
        self.LN1 = LayerNorm(76)
        self.LN2 = LayerNorm(38)
        self.LN3 = LayerNorm(2)
        self.norm = nn.LayerNorm(stand_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # fusion block
        self.transformer = Transformer(
            width=self.stand_dim,
            layers=self.transformer_layers,
            heads=self.transformer_heads
        )

    def forward(self, x):
        # MLP froward
        feature_1 = self.layer1(x)
        feature_1_relu = self.relu(feature_1)
        feature_1_dropout = self.dropout(feature_1_relu)
        feature_2 = self.layer2(feature_1_dropout)
        feature_2_relu = self.relu(feature_2)
        feature_2_dropout = self.dropout(feature_2_relu)
        feature_3 = self.layer3(feature_2_dropout)
        # print('x:', x.shape)
        # print('feature_1:', feature_1.shape)
        # print('feature_2:', feature_2.shape)
        # print('feature_3:', feature_3)
        # feature_3_softmax = self.softmax(feature_3)

        # feature_vector to stand size
        # stand_feature_0 = self.sigmoid(self.to_stand_feature_0(self.LN0(x)))
        # stand_feature_1 = self.sigmoid(self.to_stand_feature_1(self.LN1(feature_1)))
        # stand_feature_2 = self.sigmoid(self.to_stand_feature_2(self.LN2(feature_2)))
        # stand_feature_3 = self.sigmoid(self.to_stand_feature_3(self.LN3(feature_3)))

        stand_feature_0 = self.sigmoid(self.to_stand_feature_0(x))
        stand_feature_1 = self.sigmoid(self.to_stand_feature_1(feature_1))
        stand_feature_2 = self.sigmoid(self.to_stand_feature_2(feature_2))
        stand_feature_3 = self.sigmoid(self.to_stand_feature_3(feature_3))
        # print('stand_feature_0:',stand_feature_0.shape)

        # cat the features and reshape the size: (batch,layers,stand_dim)
        stand_feature_cat = torch.cat((stand_feature_0, stand_feature_1, stand_feature_2, stand_feature_3), 1) #seq_length:句子的长度   d_model:是每一个单词本来的词向量长度
        # print('stand_feature_cat:', stand_feature_cat)
        stand_feature_cat = stand_feature_cat.reshape(-1, 4, self.stand_dim)  # (batch,layers+1,stand_dim) = (batch_size, seq_length, d_model)
        # print('stand_feature_cat_after_reshape:', stand_feature_cat)

        # feature fusion
        features_after_transformer = self.transformer(stand_feature_cat)
        # print('features_after_transformer:', features_after_transformer)

        # # # layer_norm
        # features_after_transformer = self.norm(features_after_transformer)  # B L C   torch.Size([6, 4, 32])
        # features_after_transformer = self.avgpool(features_after_transformer.transpose(1, 2))  # B C 1   torch.Size([6, 32, 1])

        # reshape the features size:(batch,(layers+1)*stand_dim)
        features = features_after_transformer.reshape(-1,4*self.stand_dim)
        # features = torch.flatten(features_after_transformer,start_dim=1, end_dim=-1)
        # print('features:', features.shape)

        # add the relu
        features = self.relu(features)

        # output
        output = self.FC(features)
        # output = self.softmax(output)
        # print('output:', output)

        return output



if __name__ == '__main__':
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
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))