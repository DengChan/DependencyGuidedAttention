# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

from utils import constant, torch_utils
from model.tree import head_to_tree, tree_to_adj

MAX_SEQ_LEN = 300


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SelfAttention(nn.Module):
    def __init__(self, opt, input_dim):
        super(SelfAttention, self).__init__()
        if opt["hidden_dim"] % opt["num_heads"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (opt["hidden_dim"], opt["num_heads"]))

        self.opt = opt
        self.input_dim = input_dim
        self.num_attention_heads = opt["num_heads"]
        self.attention_head_size = int(opt["hidden_dim"] / self.num_attention_heads)
        # self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.all_head_size = opt["hidden_dim"]
        # 正常情况下 input_dim 应该等于 opt["hidden_dim"],
        # 这是为了处理直接将word emmbeddings输入 导致的维度的问题
        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

        self.dropout = nn.Dropout(opt["input_dropout"])

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
    def __init__(self, opt):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(opt["hidden_dim"], opt["feedforward_dim"])
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SelfOutput(nn.Module):
    def __init__(self, opt):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(opt["feedforward_dim"], opt["hidden_dim"])
        self.LayerNorm = nn.LayerNorm(opt["hidden_dim"])
        self.dropout = nn.Dropout(opt["input_dropout"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, opt, input_dim):
        super(EncoderLayer, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.self = SelfAttention(opt, input_dim)
        self.intermediate = Intermediate(opt)
        self.output = SelfOutput(opt)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, attn = self.self(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn


class InputLayer(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'], padding_idx=constant.PAD_ID)
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID)
        #self.dist_emb = Embedding(100, opt["dist_dim"], padding_idx=constant.PAD_ID) if opt['dist_dim'] > 0 else None
        self.input_dim = opt["emb_dim"] + opt["pos_dim"] + opt["ner_dim"]

        self.position_embeddings = nn.Embedding(MAX_SEQ_LEN, self.input_dim)
        self.init_embeddings()
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.LayerNorm = nn.LayerNorm(self.input_dim)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
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
        input_shape = words.size()
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        batch = input_shape[0]

        # embedding 拼接
        word_embs = self.emb(words)
        embs = [word_embs]
        pos_embs = self.pos_emb(pos)
        embs += [pos_embs]
        ner_embs = self.ner_emb(ner)
        embs += [ner_embs]
        embs = torch.cat(embs, dim=2)

        position_ids = torch.arange(input_shape[1], dtype=torch.long, device=torch.device("cuda"))
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        embeddings = embs + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.in_drop(embeddings)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, maxlen)

        dist_embs = None

        dep_mask = get_mask_from_adj(adj)
        # pad_mask = get_attn_pad_mask(words, words)
        pad_mask, seq_mask = get_attn_masks(torch.Tensor(l).long().cuda(), int(maxlen))
        return embeddings, dist_embs, dep_mask, pad_mask, seq_mask, adj


class Encoder(nn.Module):
    def __init__(self, opt, input_dim):
        super(Encoder, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList()
        for i in range(opt["num_layers"]):
            if i == 0:
                self.layers.append(EncoderLayer(opt, input_dim))
            else:
                self.layers.append(EncoderLayer(opt, opt["hidden_dim"]))

    def forward(self, inputs, attn_mask):
        enc_self_attns = []
        seq_inputs = inputs
        for layer in self.layers:
            seq_outputs, seq_self_attn = layer(seq_inputs, attn_mask)
            seq_inputs = seq_outputs
            enc_self_attns.append(seq_self_attn)
        return seq_inputs, enc_self_attns


class DGAModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(DGAModel, self).__init__()
        self.opt = opt
        self.input_layer = InputLayer(opt, emb_matrix)

        self.Encoder = Encoder(opt, self.input_layer.input_dim)
        # add dist dim as hidden dim
        hidden_dim = opt["hidden_dim"]
        #self.DEP_Encoder = Encoder(opt, hidden_dim, hidden_dim)
        # output mlp layers
        input_dim = hidden_dim * 3
        self.out_mlp = nn.Linear(input_dim, hidden_dim)
        self.out_mlp_2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, len(constant.LABEL_TO_ID))

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack

        embs, dist_embs, dep_mask, pad_mask, seq_mask, adj = self.input_layer(inputs)

        # Context Encoder
        seq_outputs, seq_self_attns = self.Encoder(embs, dep_mask)
        # Dependency Encoder
        # seq_outputs, dep_self_attns = self.DEP_Encoder(seq_outputs, dep_mask)
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        pool_type = self.opt['pooling']

        seq_subj_out = pool(seq_outputs, subj_mask, type=pool_type)
        seq_obj_out = pool(seq_outputs, obj_mask, type=pool_type)
        seq_h_out = pool(seq_outputs, pool_mask, type=pool_type)
        outputs = torch.cat([seq_h_out, seq_subj_out, seq_obj_out], 1)
        outputs = torch.relu(self.out_mlp(outputs))
        outputs = torch.relu(self.out_mlp_2(outputs))
        scores = self.classifier(outputs)
        # hout 用于pooling 的l2正则
        return scores, seq_h_out


def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, maxlen):
    head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
    trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
    adj = np.concatenate(adj, axis=0)
    adj = torch.from_numpy(adj)
    return Variable(adj.cuda())


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
