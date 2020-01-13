import torch
import torch.nn as nn
import math
import numpy as np
import warnings


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
        # K: [B x L x hidden size]
        Q = self.W_Q(Q)
        K = self.W_K(K)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = scores + attn_mask.float() * -1e9
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
        self.LayerNorm = nn.LayerNorm(feedforward_dim)

    def forward(self, hidden_states):
        _hidden_states = self.dense(hidden_states)
        _hidden_states = self.intermediate_act_fn(_hidden_states)
        _hidden_states = self.LayerNorm(_hidden_states)
        return _hidden_states


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


class PlainAttnLayer2(nn.Module):
    def __init__(self, input_dim, attn_dim, feedforward_dim, hidden_dim, v_dim, dropout_prob=0.0):
        super(PlainAttnLayer2, self).__init__()
        if v_dim != hidden_dim:
            warnings.warn("v dim not equal to dga hidden dim, can't use ResNet.")
        self.self = PlainSelfLayer(input_dim, attn_dim, v_dim, dropout_prob)
        self.inter = PlainIntermediate(v_dim, hidden_dim)
        #self.output = PlainOutput(feedforward_dim, hidden_dim, dropout_prob)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        self_output, scores = self.self(Q, K, V, attn_mask)
        inter_output = self.inter(self_output)
        #context = self.output(inter_output, self_output)
        return inter_output, scores


class PlainAttnLayer(nn.Module):
    def __init__(self, input_dim, attn_dim, hidden_dim, v_dim, dropout_prob=0.0):
        super(PlainAttnLayer, self).__init__()
        self.d_k = attn_dim
        self.d_v = attn_dim
        self.W_Q = nn.Linear(input_dim, attn_dim)
        self.W_K = nn.Linear(input_dim, attn_dim)
        self.W_V = nn.Linear(input_dim, hidden_dim)
        self.dropout_prob = dropout_prob
        self.layerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : # [B x L x 1]
        scores = scores + attn_mask.float() * -1e9
        # scores = scores.transpose(-1, -2)  # [B x L x L]
        attn = nn.Softmax(dim=-1)(scores)
        # if self.dropout_prob > 0.0:
        #     attn = self.dropout(attn)
        WV = gelu(self.W_V(inputs))
        context = torch.matmul(attn, WV)
        # context = self.layerNorm(context + WV)
        return context, attn


class Encoder(nn.Module):
    def __init__(self, num_layers, input_dim,
                 attn_dim, v_dim, hidden_dim, feedforward_dim,
                 dropout_prob):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout_prob)
        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.forward_layers.append(PlainAttnLayer(input_dim, attn_dim, hidden_dim,
                                                          v_dim, dropout_prob))
                self.backward_layers.append(PlainAttnLayer(input_dim, attn_dim, hidden_dim,
                                                           v_dim, dropout_prob))

            else:
                self.forward_layers.append(PlainAttnLayer(hidden_dim, attn_dim, hidden_dim,
                                                          v_dim, dropout_prob))
                self.backward_layers.append(PlainAttnLayer(hidden_dim, attn_dim, hidden_dim,
                                                           v_dim, dropout_prob))

    def forward(self, forward_inputs, backward_inputs, forward_mask, backward_mask):
        enc_self_attns = []
        for i in range(len(self.forward_layers)):
            forward_outputs, forward_self_scores = self.forward_layers[i](
                forward_inputs, forward_mask)
            backward_outputs, backward_self_scores = self.backward_layers[i](
                backward_inputs, backward_mask)

            # forward_inputs = torch.cat([forward_outputs, backward_outputs], -1)
            # backward_inputs = torch.cat([forward_outputs, backward_outputs], -1)
            forward_inputs = forward_outputs + backward_outputs
            backward_inputs = forward_outputs + backward_outputs
            if i != len(self.forward_layers)-1:
                forward_inputs = self.drop(forward_inputs)
                backward_inputs = self.drop(backward_inputs)
            enc_self_attns.append([forward_self_scores])
        return forward_inputs, backward_inputs, enc_self_attns


class BiDGAModel(nn.Module):
    def __init__(self, num_layers, input_dim,
                 attn_dim, v_dim, hidden_dim, feedforward_dim,
                 dropout_prob):
        super(BiDGAModel, self).__init__()
        self.Encoder = Encoder(num_layers, input_dim,
                               attn_dim, v_dim, hidden_dim, feedforward_dim,
                               dropout_prob)

    def forward(self, forward_inputs, backward_inputs, forward_mask, backward_mask):
        f_outputs, b_outputs, self_attns = self.Encoder(forward_inputs, backward_inputs,
                                                        forward_mask, backward_mask)
        outputs = f_outputs
        return outputs, self_attns
