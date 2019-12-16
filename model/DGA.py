# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import numpy as np


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attn_dim, num_heads, dropout_prob=0.0):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_attention_heads = num_heads
        self.attention_head_size = attn_dim
        # self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.all_head_size = attn_dim * num_heads
        # 正常情况下 input_dim 应该等于 opt["hidden_dim"],
        # 这是为了处理直接将word emmbeddings输入 导致的维度的问题
        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1).float() * -10000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


class Intermediate(nn.Module):
    def __init__(self, attn_dim, feedforward_dim):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(attn_dim, feedforward_dim)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SelfOutput(nn.Module):
    def __init__(self, feedforward_dim, hidden_dim, dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(feedforward_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, hidden_dim, feedforward_dim, num_heads, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.input_dim = input_dim
        self.self = SelfAttention(input_dim, attn_dim, num_heads, dropout_prob)
        self.intermediate = Intermediate(attn_dim, feedforward_dim)
        self.output = SelfOutput(feedforward_dim, hidden_dim, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, attn = self.self(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn


class PlainAttnLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, hidden_dim, v_dim, dropout_prob=0.0):
        super(PlainAttnLayer, self).__init__()
        self.d_k = attn_dim
        self.d_v = attn_dim
        self.W_Q = nn.Linear(input_dim, attn_dim)
        self.W_K = nn.Linear(input_dim, attn_dim)
        self.W_V = nn.Linear(input_dim, v_dim)
        self.dropout_prob = dropout_prob
        self.W_out = nn.Linear(v_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        Q = self.W_Q(Q)
        K = self.W_K(K)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : # [B x L x 1]
        scores = scores + attn_mask.float() * -1e9
        scores = scores.transpose(-1, -2)  # [B x L x L]
        attn = nn.Softmax(dim=-1)(scores)
        if self.dropout_prob > 0.0:
            attn = self.dropout(attn)
        WV = self.W_V(V)
        context = torch.matmul(attn, WV) # [B x L x E]
        context = gelu(self.W_out(context))
        context = self.layerNorm(context)
        return context, attn


class Encoder(nn.Module):
    def __init__(self, num_layers, attn_dim, input_dim, hidden_dim, feedforward_dim, num_heads, dropout_prob):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # self.layers.append(EncoderLayer(input_dim, attn_dim,
                #                                 hidden_dim, feedforward_dim, num_heads, dropout_prob))
                self.layers.append(PlainAttnLayer(input_dim, attn_dim,
                                                  hidden_dim, feedforward_dim, dropout_prob))
            else:
                # self.layers.append(EncoderLayer(hidden_dim, attn_dim,
                #                                 hidden_dim, feedforward_dim, num_heads, dropout_prob))
                self.layers.append(PlainAttnLayer(hidden_dim, attn_dim,
                                                  hidden_dim, feedforward_dim, dropout_prob))

    def forward(self, inputs, attn_mask):
        enc_self_attns = []
        seq_inputs = inputs
        for layer in self.layers:
            seq_outputs, seq_self_attn = layer(seq_inputs, attn_mask)
            seq_inputs = seq_outputs
            enc_self_attns.append(seq_self_attn)
        return seq_inputs, enc_self_attns


class DGAModel(nn.Module):
    def __init__(self, num_layers, input_dim, attn_dim, hidden_dim, feedforward_dim, num_heads, dropout_prob):
        super(DGAModel, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.Encoder = Encoder(num_layers, input_dim, attn_dim, hidden_dim, feedforward_dim, num_heads, dropout_prob)

    def forward(self, inputs, dep_mask):
        seq_outputs, self_attns = self.Encoder(inputs, dep_mask)
        return seq_outputs, self_attns