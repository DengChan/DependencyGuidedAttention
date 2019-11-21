# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import math

from utils import constant, torch_utils
from model.tree import Tree, head_to_tree, tree_to_adj

from model.bert import BertConfig, BertForTokenClassification, BertTokenizer, BertModel

MAX_SEQ_LEN = 100


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def PoolingLayer(inputs, pool_mask, subj_mask, obj_mask, pool_type):
    h_out = pool(inputs, pool_mask, type=pool_type)
    subj_out = pool(inputs, subj_mask, type=pool_type)
    obj_out = pool(inputs, obj_mask, type=pool_type)
    outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
    return outputs, h_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, opt):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = opt["K_dim"]

    def forward(self, Q, K, V, attn_mask, help_scores=None, ret_scores=False):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        if ret_scores:
            return context, scores
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.opt = opt
        self.d_k = opt["K_dim"]
        self.d_v = opt["V_dim"]
        self.n_heads = opt["num_heads"]
        self.W_Q = nn.Linear(input_dim, self.d_k * self.n_heads)
        self.W_K = nn.Linear(input_dim, self.d_k * self.n_heads)
        self.W_V = nn.Linear(input_dim, self.d_v * self.n_heads)
        self.DotProductAttention = ScaledDotProductAttention(opt)
        self.Wout = nn.Linear(self.n_heads * self.d_v, self.output_dim)
        self.layerNorm = nn.LayerNorm(self.output_dim)

    def forward(self, Q, K, V, attn_mask, help_scores=None, ret_scores=False):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        try:
            q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        except:
            print("ERROR")
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.DotProductAttention(q_s, k_s, v_s, attn_mask, help_scores, ret_scores)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.Wout(context)
        if residual.size(-1) != output.size(-1):
            return self.layerNorm(output), attn
        return self.layerNorm(output + residual), attn # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, feedforward_dim, dropout_prob):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.feedforward_dim = feedforward_dim
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.feedforward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.feedforward_dim, out_channels=self.input_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(self.input_dim)

    def forward(self, inputs):
        residual = self.drop(inputs) # inputs : [batch_size, len_q, d_model]
        output = gelu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layerNorm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(EncoderLayer, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enc_self_attn = MultiHeadAttention(opt, input_dim, output_dim)
        self.pos_ffn = PoswiseFeedForwardNet(output_dim, opt["feedforward_dim"], opt["input_dropout"])

    def forward(self, enc_inputs, enc_self_attn_mask, help_scores=None, ret_scores=False):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, help_scores, ret_scores) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class LSTMLayer(nn.Module):
    def __init__(self, opt, in_dim, hidden_dim):
        super(LSTMLayer, self).__init__()
        self.opt = opt
        self.hidden_dim = hidden_dim
        input_size = in_dim
        self.rnn = nn.LSTM(input_size, hidden_dim, opt['rnn_layers'], batch_first=True,
                           dropout=opt['rnn_dropout'], bidirectional=True)
        self.in_dim = hidden_dim * 2
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        # self.rnn_layer_norm = nn.LayerNorm(self.in_dim)

    def forward(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = self.rnn_zero_state(batch_size, self.hidden_dim, self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        rnn_outputs = self.rnn_drop(rnn_outputs)
        return rnn_outputs

    def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
        total_layers = num_layers * 2 if bidirectional else num_layers
        state_shape = (total_layers, batch_size, hidden_dim)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0


class InputLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.pos_emb = Embedding(len(constant.POS_TO_ID), opt['pos_dim'], padding_idx=constant.PAD_ID)
        self.ner_emb = Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack
        # padding 是1 词是0
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        batch = len(words)

        pos_embs = self.pos_emb(pos)
        ner_embs = self.ner_emb(ner)

        adj, dists = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, deprel, maxlen)

        dist_embs = None

        dep_mask = get_mask_from_adj(adj)
        # pad_mask = get_attn_pad_mask(words, words)
        pad_mask, seq_mask = get_attn_masks(torch.Tensor(l).long().cuda(), int(maxlen))
        return pos_embs, ner_embs, dist_embs, dep_mask, pad_mask, seq_mask, adj


class Encoder(nn.Module):
    def __init__(self, opt, config):
        super(Encoder, self).__init__()
        self.opt = opt
        self.config = config
        self.bert = BertModel(config)
        self.load_bert_model()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_bert_model(self):
        self.bert.from_pretrained(self.opt["model_name_or_path"],
                                  config=self.config,
                                  cache_dir=self.opt["cache_dir"] if self.opt["cache_dir"] else None)

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        :return: outputs[0] : 最后一层的输出 [Batch X Length X Hidden Size]
                outputs[1] : 每一个元素是每一层的输出[Batch X Length X Hidden Size] 的 list
                outputs[2]: 每一层Attention值[Batch X Heads number X Length X Length]的 list
        """

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        outputs = (outputs[0],) + outputs[2:]  # add hidden states and attention if they are here
        return outputs


class Decoder(nn.Module):
    def __init__(self, opt, config):
        super(Decoder, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, len(constant.LABEL_TO_ID))

    def forward(self, seq_outputs, subj_pos, obj_pos, adj):
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        if self.opt["deprel_edge"]:
            adj_tmp = adj.eq(0).eq(0).long()
            pool_mask = (adj_tmp.sum(2) + adj_tmp.sum(1)).eq(0).unsqueeze(2)
        else:
            pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        pool_type = self.opt['pooling']
        seq_subj_out = pool(seq_outputs, subj_mask, type=pool_type)
        seq_obj_out = pool(seq_outputs, obj_mask, type=pool_type)
        seq_h_out = pool(seq_outputs, pool_mask, type=pool_type)
        outputs = torch.cat([seq_subj_out, seq_obj_out, seq_h_out], -1)
        outputs = self.dropout(outputs)
        outputs = gelu(self.out_mlp(outputs))
        logits = self.classifier(outputs)
        return logits, seq_h_out


class DGAModel(nn.Module):
    def __init__(self, opt, config):
        super(DGAModel, self).__init__()
        self.opt = opt

        self.Encoder = Encoder(opt, config)
        self.Decoder = Decoder(opt, config)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = None
        self.rel_logits = None

    def forward(self, inputs):
        input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, subj_poses, obj_poses, bg_list, ed_list, subj_type_ids, obj_type_ids = inputs  # unpack

        self.labels = label_ids

        # Encoder
        seq_outputs = self.Encoder(input_ids, input_masks, segment_ids)[0]

        self.rel_logits, seq_h_out = self.Decoder(seq_outputs, subj_poses, obj_poses, adjs)
        rel_loss = self.cal_rel_loss(self.rel_logits, label_ids)

        return rel_loss, seq_h_out

    def cal_rel_loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def predict(self):
        probs = F.softmax(self.rel_logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(self.rel_logits.data.cpu().numpy(), axis=1).tolist()
        return predictions, probs

    def get_labels(self):
        # lazy transform to list
        self.labels = self.labels.cpu().numpy().tolist()
        return self.labels


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).cuda()


def get_mask_from_adj(adj):

    mask = adj.eq(0)
    return mask.cuda()


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k).cuda()  # batch_size x len_q x len_k


def get_attn_masks(lengths, slen):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)

    alen = torch.arange(slen, dtype=torch.long)
    if torch.cuda.is_available():
        alen = alen.cuda()
    mask = alen < lengths[:, None]
    mask = mask.eq(0).unsqueeze(1).repeat(1, slen, 1)
    attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    attn_mask = attn_mask.eq(0)
    # sanity check
    assert attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
