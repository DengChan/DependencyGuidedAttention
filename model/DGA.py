# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import copy

from utils import constant, torch_utils
from model.tree import Tree, head_to_tree, tree_to_adj

MAX_SEQ_LEN = 300


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


class InputLayer(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = Embedding(len(constant.POS_TO_ID), opt['pos_dim'], padding_idx=constant.PAD_ID) if opt['pos_dim'] > 0 else None
        self.ner_emb = Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID) if opt['ner_dim'] is not None else None

        self.input_dim = opt["emb_dim"] + opt["pos_dim"] + opt["ner_dim"]

        self.position_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(MAX_SEQ_LEN, self.input_dim), freeze=True)
        self.init_embeddings()
        self.in_drop = nn.Dropout(opt['input_dropout'])

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb_matrix[0] = 0.0
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack
        # padding 是1 词是0
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        batch = len(words)

        # embedding 拼接
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] is not None:
            pos_embs = self.pos_emb(pos)
            embs += [pos_embs]
        if self.opt['ner_dim'] is not None:
            ner_embs = self.ner_emb(ner)
            embs += [ner_embs]
        embs = torch.cat(embs, dim=2)

        position = [i for i in range(1, maxlen+1)]
        position = torch.LongTensor(position).unsqueeze(0).repeat(batch, 1).cuda()

        embs = embs + self.position_emb(position)
        embs = self.in_drop(embs.cuda())

        adj, adj_r = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, deprel, maxlen)
        if self.opt["directed"]:
            adj = adj + adj_r

        dep_mask = get_mask_from_adj(adj)
        # pad_mask = get_attn_pad_mask(words, words)
        pad_mask, seq_mask = get_attn_masks(torch.Tensor(l).long().cuda(), int(maxlen))
        return embs, dep_mask, pad_mask, seq_mask, adj


class ScaledDotProductAttention(nn.Module):
    def __init__(self, opt):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = opt["K_dim"]

    def forward(self, Q, K, V, attn_mask, help_scores=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.

        if help_scores is not None:
            help_scores_mask = help_scores > -1e8
            help_scores_mask_dep = help_scores_mask.float() * 0.5
            help_scores_mask_seq = help_scores_mask.float() * 0.5 + help_scores_mask.int().eq(0).float()
            scores = help_scores * help_scores_mask_dep + scores*help_scores_mask_seq

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
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

    def forward(self, Q, K, V, attn_mask, help_scores=None):
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
        context, attn = self.DotProductAttention(q_s, k_s, v_s, attn_mask, help_scores)
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

    def forward(self, enc_inputs, enc_self_attn_mask, help_scores=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, help_scores) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class HyperLayer(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(HyperLayer, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.SeqEncoderLayer = EncoderLayer(opt, input_dim, output_dim)
        self.DepEncoderLayer = EncoderLayer(opt, input_dim, output_dim)

    def forward(self, seq_inputs, dep_inputs,  seq_attn_mask, dep_attn_mask):
        dep_outputs, dep_self_attn = self.DepEncoderLayer(dep_inputs, dep_attn_mask)
        seq_outputs, seq_self_attn = self.SeqEncoderLayer(seq_inputs, seq_attn_mask, dep_self_attn)
        return seq_outputs, seq_self_attn


class Encoder(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList()
        for i in range(opt["num_layers"]):
            if i == 0:
                self.layers.append(HyperLayer(opt, input_dim, output_dim))
            else:
                self.layers.append(HyperLayer(opt, output_dim, output_dim))

    def forward(self, input, seq_attn_mask, dep_attn_mask):
        enc_self_attns = []
        enc_outputs = input
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, seq_attn_mask, dep_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Scorer(nn.Module):
    def __init__(self, opt):
        super(Scorer, self).__init__()
        self.label_num = len(constant.LABEL_TO_ID)
        self.label_embs = Embedding(self.label_num, opt["label_dim"])

    def forward(self, features):
        """

        :param features: B X E
        :return:
        """
        # L X E
        label_embedding = self.label_embs(torch.arange(self.label_num).cuda())
        scores = torch.matmul(features, label_embedding.transpose(0, 1))
        # scores = torch.sigmoid(scores)
        return scores


class DGAModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(DGAModel, self).__init__()
        self.opt = opt
        self.input_layer = InputLayer(opt, emb_matrix)

        self.Encoder = Encoder(opt, self.input_layer.input_dim, opt["hidden_dim"])

        # output mlp layers
        input_dim = opt["hidden_dim"] * 3
        self.out_mlp = nn.Linear(input_dim, opt["hidden_dim"])
        self.classifier = nn.Linear(opt["hidden_dim"], len(constant.LABEL_TO_ID))
        self.scorer = Scorer(opt)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack

        embs, dep_mask, pad_mask, seq_mask, adj = self.input_layer(inputs)
        seq_enc_outputs, seq_dec_enc_attns = self.Encoder(embs, seq_mask, dep_mask)
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        pool_type = self.opt['pooling']

        subj_out = pool(seq_enc_outputs, subj_mask, type=pool_type)
        obj_out = pool(seq_enc_outputs, obj_mask, type=pool_type)
        h_out = pool(seq_enc_outputs, pool_mask, type=pool_type)

        outputs = torch.cat([subj_out, obj_out, h_out], -1)
        outputs = gelu(self.out_mlp(outputs))
        scores = self.classifier(outputs)
        # hout 用于pooling 的l2正则
        return scores, outputs


def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, deprel=None, maxlen=100):
    head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
    if deprel is not None:
        deprel = deprel.cpu().numpy()
        trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i], deprel[i]) for i in range(len(l))]
    else:
        trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
    # adj 邻接边为边类型
    adjs = []
    adjs_r = []
    for tree in trees:
        adj = tree_to_adj(maxlen, tree, directed=True, edge_info=True)
        adj_r = copy.deepcopy(adj.T)
        adj_r[adj_r>1] += constant.DEPREL_COUNT
        adjs.append(adj.reshape(1, maxlen, maxlen))
        adjs_r.append(adj_r.reshape(1, maxlen, maxlen))
    adjs = np.concatenate(adjs, axis=0)
    adjs_r = np.concatenate(adjs_r, axis=0)
    adjs = torch.from_numpy(adjs)
    adjs_r = torch.from_numpy(adjs_r)
    adjs = Variable(adjs.cuda())
    adjs_r = Variable(adjs_r.cuda())

    return adjs, adjs_r


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
    B = adj.size(0)
    N = adj.size(1)

    I = torch.eye(N).float().cuda()
    I = I.unsqueeze(0).repeat(B, 1, 1)
    adj = adj + I

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