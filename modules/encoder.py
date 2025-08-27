import torch.nn as nn

from modules.common import LayerNorm, SublayerConnection
from utils import utils
from utils.utils import clones


class Encoder(nn.Module):
    def __init__(self, layer, N, PAM):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N

    def forward(self, x, mask):
        s=[]
        for i,layer in enumerate(self.layers):
            x = layer(x, mask)
            s.append(self.PAM[i](x))


        o = s[0]
        for i in range(1,len(s)):
            o +=  s[i]
        return o


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)