from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention_model import AttModel
from modules.common import subsequent_mask, LayerNorm
from modules.decoder import DecoderLayer, Decoder
from modules.encoder import Encoder, EncoderLayer
from utils.utils import pad_tokens, pack_wrapper, clones


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed


    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # print(f'src: {src.shape}')
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PAM(nn.Module):
    def __init__(self, dim=512):
        super(PAM, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 13, 1, 13//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        B, H, C = x.shape
        # print(f'B, H, C : {(B, H, C)}')
        assert int(math.sqrt(H))**2==H, f'{x.shape}'
        cnn_feat = x.transpose(1, 2).view(B, C, int(math.sqrt(H)), int(math.sqrt(H))).contiguous()
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class EncoderDecoder(AttModel):

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        tgt_vocab = self.vocab_size + 1

        self.embeded = Embeddings(args.d_vf, tgt_vocab)
        self.model = self.__build_model(tgt_vocab)
        self.__init_model()

        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.logit_mesh = nn.Linear(args.d_model, args.d_model)

    def __build_model(self, tgt_vocab):
        attn = MultiHeadedAttention(self.num_heads, self.d_model, dropout=self.dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        pp = PAM(self.d_model)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, deepcopy(attn), deepcopy(ff), self.dropout), self.num_layers, pp),
            Decoder(
                DecoderLayer(self.d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), self.dropout),
                self.num_layers),
            LayerNorm(self.d_model),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), deepcopy(position))
        )
        return model


    def init_hidden(self, bsz):
        return []

    def __init_model(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _prepare_feature(self, fc_feats, att_feats, att_masks, meshes=None):
        att_feats = pad_tokens(att_feats)
        att_feats, seq, _, att_masks, seq_mask, _ = self._prepare_feature_forward(att_feats, att_masks, meshes)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_mesh(self, att_feats, att_masks=None, meshes=None):
        att_feats = pad_tokens(att_feats)
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None

        return att_feats, meshes, att_masks, meshes_mask

    def _prepare_feature_forward(self, att_feats, att_masks=None, meshes=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None

        return att_feats, seq, meshes, att_masks, seq_mask, meshes_mask

    def _forward(self, fc_feats, att_feats, report_ids, att_masks=None):
        # log_message(fc_feats, att_feats, report_ids, att_masks)
        att_feats, report_ids, att_masks, report_mask = self._prepare_feature_mesh(att_feats, att_masks, report_ids)
        out = self.model(att_feats, report_ids, att_masks, report_mask)

        # print(f'out: {out}')
        outputs = F.log_softmax(self.logit(out), dim=-1)
        # print(f'outputs: {outputs}')

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.long().unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

    def _encode(self, fc_feats, att_feats, att_masks=None):

        att_feats, _, att_masks, _ = self._prepare_feature_mesh(att_feats, att_masks)
        out = self.model.encode(att_feats, att_masks)
        return out