# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import numpy as np
import warnings


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
        assert attn_dim * num_heads == hidden_dim
        self.input_dim = input_dim
        self.self = SelfAttention(input_dim, attn_dim, num_heads, dropout_prob)
        self.intermediate = Intermediate(attn_dim*num_heads, feedforward_dim)
        self.output = SelfOutput(feedforward_dim, hidden_dim, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, attn = self.self(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn


class PlainSelfLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, v_dim, dropout_prob):
        super(PlainSelfLayer, self).__init__()
        self.input_dim = input_dim
        self.d_k = attn_dim
        self.d_v = attn_dim
        self.W_Q = nn.Linear(input_dim, attn_dim)
        self.W_K = nn.Linear(input_dim, attn_dim)
        self.W_V = nn.Linear(input_dim, v_dim)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

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
        selfOutput = torch.matmul(attn, WV)  # [B x L x E]
        return selfOutput, attn


class PlainIntermediate(nn.Module):
    def __init__(self, v_dim, feedforward_dim):
        super(PlainIntermediate, self).__init__()
        self.dense = nn.Linear(v_dim, feedforward_dim)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class PlainOutput(nn.Module):
    def __init__(self, feedforward_dim, hidden_dim, dropout_prob):
        super(PlainOutput, self).__init__()
        self.dense = nn.Linear(feedforward_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if input_tensor.size()[-1] != hidden_states.size()[-1]:
            hidden_states = self.LayerNorm(hidden_states)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PlainAttnLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, feedforward_dim, hidden_dim, v_dim, dropout_prob=0.0):
        super(PlainAttnLayer, self).__init__()
        if v_dim != hidden_dim:
            warnings.warn("v dim not equal to dga hidden dim, can't use ResNet.")
        self.self = PlainSelfLayer(input_dim, attn_dim, v_dim, dropout_prob)
        self.inter = PlainIntermediate(v_dim, feedforward_dim)
        self.output = PlainOutput(feedforward_dim, hidden_dim, dropout_prob)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        self_output, attn = self.self(Q, K, V, attn_mask)
        inter_output = self.inter(self_output)
        context = self.output(inter_output, self_output)
        return context, attn


class UnitAttnLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, feedforward_dim, hidden_dim, v_dim, dropout_prob=0.0):
        super(UnitAttnLayer, self).__init__()
        if v_dim*2 != hidden_dim:
            warnings.warn("(2 x v dim) not equal to dga hidden dim, can't use ResNet.")
        self.seq_self = PlainSelfLayer(input_dim, attn_dim, v_dim, dropout_prob)
        self.seq_inter = PlainIntermediate(v_dim, feedforward_dim)
        self.seq_output = PlainOutput(feedforward_dim, hidden_dim, dropout_prob)

        self.dep_self = PlainSelfLayer(input_dim, attn_dim, v_dim, dropout_prob)
        self.dep_inter = PlainIntermediate(v_dim, feedforward_dim)
        self.dep_output = PlainOutput(feedforward_dim, hidden_dim, dropout_prob)

    def forward(self, Q,  K, V, seq_mask, dep_mask):
        seq_self, seq_attn = self.seq_self(Q, K, V, seq_mask)
        seq_inter = self.seq_inter(seq_self)
        seq_output = self.seq_output(seq_inter, seq_self)

        dep_self, dep_attn = self.dep_self(Q, K, V, dep_mask)
        dep_inter = self.dep_inter(dep_self)
        dep_output = self.dep_output(dep_inter, dep_self)
        context = torch.cat([seq_output, dep_output], -1)
        return context, dep_attn


class Encoder(nn.Module):
    def __init__(self, num_layers, seq_input_dim, dep_input_dim,
                 attn_dim, num_heads, hidden_dim, feedforward_dim,
                 dropout_prob):
        super(Encoder, self).__init__()
        self.seq_layers = nn.ModuleList()
        self.dep_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.seq_layers.append(EncoderLayer(seq_input_dim, attn_dim,
                                                    hidden_dim, feedforward_dim,
                                                    num_heads, dropout_prob))
                self.dep_layers.append(EncoderLayer(dep_input_dim, attn_dim,
                                                    hidden_dim, feedforward_dim,
                                                    num_heads, dropout_prob))

                # self.layers.append(PlainAttnLayer(input_dim, attn_dim, feedforward_dim,
                #                                   hidden_dim, v_dim, dropout_prob))
            else:
                self.seq_layers.append(EncoderLayer(hidden_dim, attn_dim,
                                                    hidden_dim, feedforward_dim,
                                                    num_heads, dropout_prob))
                self.dep_layers.append(EncoderLayer(hidden_dim, attn_dim,
                                                    hidden_dim, feedforward_dim,
                                                    num_heads, dropout_prob))

                # self.layers.append(PlainAttnLayer(hidden_dim, attn_dim, feedforward_dim,
                #                                   hidden_dim, v_dim, dropout_prob))

    def forward(self, seq_inputs, dep_inputs, attn_mask, dep_mask):
        enc_self_attns = []
        for i in range(len(self.dep_layers)):
            seq_outputs, seq_self_attn = self.seq_layers[i](seq_inputs, attn_mask)
            # dep_outputs, dep_self_attn = self.dep_layers[i](dep_inputs, dep_mask)
            # seq_inputs = torch.cat([seq_outputs, dep_outputs], -1)
            # dep_inputs = torch.cat([seq_outputs, dep_outputs], -1)
            # dep_inputs = dep_outputs
            seq_inputs = seq_outputs
            enc_self_attns.append([seq_self_attn])
        return seq_inputs, enc_self_attns


class DGAModel(nn.Module):
    def __init__(self, num_layers, seq_input_dim, dep_input_dim,
                 attn_dim, num_heads, hidden_dim, feedforward_dim,
                 dropout_prob):
        super(DGAModel, self).__init__()
        self.Encoder = Encoder(num_layers, seq_input_dim, dep_input_dim,
                               attn_dim, num_heads, hidden_dim, feedforward_dim,
                               dropout_prob)

    def forward(self, seq_inputs, dep_inputs, seq_mask, dep_mask):
        seq_outputs, self_attns = self.Encoder(seq_inputs, dep_inputs, seq_mask, dep_mask)
        return seq_outputs, self_attns


class UnitDGAModel(nn.Module):
    def __init__(self, num_layers, input_dim, attn_dim, v_dim, hidden_dim, feedforward_dim, dropout_prob):
        super(UnitDGAModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(UnitAttnLayer(input_dim,
                                                 attn_dim, feedforward_dim,
                                                 hidden_dim, v_dim, dropout_prob))
            else:

                self.layers.append(UnitAttnLayer(hidden_dim*2,
                                                 attn_dim, feedforward_dim,
                                                 hidden_dim, v_dim, dropout_prob))

    def forward(self, inputs, attn_mask, dep_mask):
        enc_self_attns = []
        seq_inputs = inputs
        for layer in self.layers:
            seq_outputs, seq_self_attn = layer(seq_inputs, seq_inputs, seq_inputs, attn_mask, dep_mask)
            seq_inputs = seq_outputs
            enc_self_attns.append(seq_self_attn)
        return seq_inputs, enc_self_attns