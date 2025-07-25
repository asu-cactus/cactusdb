"""
CREDIT: The following code is adapted from the QueryFormer work: https://github.com/zhaoyue-ntu/QueryFormer
"""
from .database_util import collator
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(
        self, in_feature=69, hid_units=256, contract=1, mid_layers=True, res_con=True
    ):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):

        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out


class FeatureEmbed(nn.Module):
    def __init__(
        self,
        embed_size=32,
        kernels=100,
        dims_length=50,
        flops_length=1,
    ):
        super(FeatureEmbed, self).__init__()

        self.embed_size = embed_size
        self.dims_length = dims_length
        self.flops_length = flops_length
        self.kernelEmbed = nn.Embedding(kernels, embed_size)

        self.kernelDimsLinear = nn.Linear(dims_length, embed_size)
        self.kernelFlopsLinear = nn.Linear(flops_length, embed_size)

        self.project = nn.Linear(embed_size * 3, embed_size * 3)

    def forward(self, feature):
        kernelId, dims, flops = torch.split(feature, (1, self.dims_length, self.flops_length), dim=-1)

        kernelEmb = self.getKernel(kernelId)
        dimsEmb = self.getKernelDims(dims)
        flopsEmb = self.getKernelFlops(flops)

        final = torch.cat((kernelEmb, dimsEmb, flopsEmb), dim=1)
        final = F.leaky_relu(self.project(final))

        return final

    def getKernel(self, kernelId):
        emb = self.kernelEmbed(kernelId.long())
        return emb.squeeze(1)

    def getKernelDims(self, dims):
        emb = F.leaky_relu(self.kernelDimsLinear(dims))
        return emb.squeeze(1)

    def getKernelFlops(self, flops):
        emb = F.leaky_relu(self.kernelFlopsLinear(flops))
        return emb.squeeze(1)


class Model2Vec(nn.Module):
    def __init__(
        self,
        emb_size=32,
        ffn_dim=32,
        head_size=8,
        dropout=0.1,
        attention_dropout_rate=0.1,
        n_layers=8,
        bin_number=50,
        pred_hid=256,
    ):

        super(Model2Vec, self).__init__()
        hidden_dim = emb_size * 3
        self.hidden_dim = hidden_dim
        self.head_size = head_size

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        self.embbed_layer = FeatureEmbed(emb_size)

        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        self.pred2 = Prediction(hidden_dim, pred_hid)

    def forward(self, batched_data):
        # attention bias, rel pos, x
        attn_bias, rel_pos, x = (
            batched_data.attn_bias,
            batched_data.rel_pos,
            batched_data.x,
        )
        # heights encoding
        heights = batched_data.heights

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(
            0, 3, 1, 2
        )  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        x_view = x.view(-1, 52)
        node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)

        # -1 is number of dummy

        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        return output[:, 0, :]

    def get_pred1(self, embeddings):
        return self.pred(embeddings)

    def get_pred2(self, embeddings):
        return self.pred2(embeddings)

    def get_embeddings(self, dataset, device):
        batch, _ = collator(list(zip(*[dataset[i] for i in range(len(dataset))])))
        with torch.no_grad():
            batch = batch.to(device)
            embeddings = self.forward(batch)
        return embeddings


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size**-0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, head_size
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
