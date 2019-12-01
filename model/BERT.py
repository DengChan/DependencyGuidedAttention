import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from utils import constant

from model.bert import BertModel
from model.GCN import GCN


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Attention(nn.Module):
    def __init__(self, K_dim, V_dim, input_Q_dim, input_K_dim):
        super(Attention, self).__init__()
        self.d_k = K_dim
        self.d_v = V_dim
        self.W_Q = nn.Linear(input_Q_dim, self.d_k)
        self.W_K = nn.Linear(input_K_dim, self.d_k)
        self.W_V = nn.Linear(input_Q_dim, self.d_v)

    def forward(self, Q, K, V, attn_mask):
        Q = self.W_Q(Q) # [B x L x E]
        K = self.W_K(K) # [B x 1 x E]
        V = self.W_V(V) # [B x L x E]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : # [B x L x 1]
        scores = scores + attn_mask.float() * -1e9
        scores = scores.transpose(-1, -2) # [B x 1 x L]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V).squeeze(1) # [B x 1 x E]
        return context, attn


class Encoder(nn.Module):
    def __init__(self, opt, config):
        super(Encoder, self).__init__()
        self.opt = opt
        self.config = config
        self.bert = BertModel(config)
        self.load_bert_model()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_bert_model(self):
        self.bert = self.bert.from_pretrained(self.opt["model_name_or_path"],
                                              config=self.config,
                                              cache_dir=self.opt["cache_dir"] if self.opt["cache_dir"] else None)
        print("Load Bert Model successfully")

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
        pos_indicator = int(self.opt["max_seq_length"])
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask
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


class EmbeddingLayer(nn.Module):
    def __init__(self, opt):
        super(EmbeddingLayer, self).__init__()
        self.label_emb = nn.Embedding(len(constant.LABEL_TO_ID), opt["label_dim"])
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt["pos_dim"], padding_idx=constant.PAD_ID)
        self.dep_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt["dep_dim"], padding_idx=constant.PAD_ID)

    def forward(self, label_ids, pos_ids, dep_ids):
        pos_embs = self.pos_emb(pos_ids)
        dep_embs = self.dep_emb(dep_ids)
        label_embs = self.label_emb(label_ids)
        return pos_embs, dep_embs, label_embs


class Decoder2(nn.Module):
    def __init__(self, opt, config):
        super(Decoder2, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)
        self.structure_decoder = GCN(config.hidden_size + opt["pos_dim"] + opt["dep_dim"],
                                     opt["gcn_hidden_dim"], opt["gcn_layers"],
                                     opt["input_dropout"], opt["input_dropout"])
        self.context_decoder = Attention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.entity_mlp = nn.Linear(config.hidden_size * 2, opt["entity_hidden_dim"])

        hidden_size = opt["entity_hidden_dim"]+opt["V_dim"]+opt["gcn_hidden_dim"]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(hidden_size, opt["hidden_dim"])
        self.out_mlp2 = nn.Linear(opt["hidden_dim"], opt["label_dim"])
        self.classifier = nn.Linear(opt["label_dim"], len(constant.LABEL_TO_ID))

    def forward(self, inputs, pos_ids, dep_ids, subj_pos, obj_pos, adj):
        bs = pos_ids.size()[0]
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask
        if self.opt["deprel_edge"]:
            adj_tmp = adj.eq(0).eq(0).long()
            pool_mask = (adj_tmp.sum(2) + adj_tmp.sum(1)).eq(0).unsqueeze(2)
        else:
            pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        pos_embs, dep_embs, label_embs = self.embs(labels, pos_ids, dep_ids)

        # context
        label_indicator = torch.mean(label_embs, 0).unsqueeze(0).repeat([bs, 1]).unsqueeze(1) # [B x 1 x label_dim]
        attn_mask = subj_pos.eq(pos_indicator) + obj_pos.eq(pos_indicator)
        attn_mask = attn_mask.unsqueeze(2) # [B x L x 1]
        context_feature, attn = self.context_decoder(inputs, label_indicator, inputs, attn_mask) # [B x V_dim]

        # structure
        adj_gcn = adj
        if self.opt["deprel_edge"]:
            adj_gcn = adj.eq(0).eq(0)
        gcn_inputs = torch.cat([inputs, pos_embs, dep_embs], -1)
        gcn_outputs = self.structure_decoder(adj_gcn, gcn_inputs) # [B x L x GCN_hidden]
        structure_feature = pool(gcn_outputs, pool_mask, pool_type) # [B x GCN_hidden]

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        obj_out = pool(inputs, obj_mask, type=pool_type)
        entity_feature = torch.cat([subj_out, obj_out], -1)
        entity_feature = gelu(self.entity_mlp(entity_feature))

        # output
        features = torch.cat([entity_feature, structure_feature, context_feature], -1)

        outputs = self.dropout(features)
        outputs = gelu(self.out_mlp(outputs))
        outputs = gelu(self.out_mlp2(outputs))
        logits = self.classifier(outputs)
        return logits, structure_feature, outputs


class BertRE(nn.Module):
    def __init__(self, opt, config):
        super(BertRE, self).__init__()
        self.opt = opt

        self.Encoder = Encoder(opt, config)
        self.Decoder = Decoder2(opt, config)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = None
        self.rel_logits = None

    def forward(self, inputs):
        input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, subj_poses, obj_poses, bg_list, ed_list, subj_type_ids, obj_type_ids = inputs  # unpack

        self.labels = label_ids

        # Encoder
        seq_outputs = self.Encoder(input_ids, input_masks, segment_ids)[0]

        self.rel_logits, seq_h_out, hidden_features = self.Decoder(seq_outputs, pos_ids, deprel_ids,
                                                                   subj_poses, obj_poses, adjs)

        rel_loss = self.cal_rel_loss(self.rel_logits, label_ids)
        match_loss = self.cal_match_loss(self.rel_logits, label_ids, hidden_features, self.Decoder.embs.label_emb)
        loss = rel_loss + self.opt["match_loss_weight"] * match_loss
        return loss, seq_h_out

    def cal_rel_loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def cal_match_loss(self, logits, labels, hidden_features, label_emb):
        # positive loss
        labels_features = label_emb(labels) # [B x E]
        # distance_loss = torch.norm(hidden_features - labels_features, dim=-1) # [B]
        distance_loss = torch.sum(hidden_features * labels_features, -1)
        score_mask = labels.eq(0)
        distance_loss = distance_loss.masked_fill_(score_mask, 1.0) # mask 负标签， mask位置经过log后变为0

        distance_loss = torch.sum(torch.sigmoid(distance_loss).log(), -1)

        # negtive loss
        second_labels = torch.argmax(logits[:, 1:], -1) + 1
        second_labels_features = label_emb(second_labels)
        second_distance_loss = torch.sum(hidden_features * second_labels_features, -1)
        second_score_mask = second_labels.eq(0).eq(0)
        second_distance_loss = second_distance_loss.masked_fill_(second_score_mask, -1.0)  # mask正标签, mask位置经过log后变为0
        second_distance_loss = torch.sum(torch.sigmoid(-1 * second_distance_loss).log(), -1)
        return (distance_loss+second_distance_loss) * -1

    def predict(self):
        probs = F.softmax(self.rel_logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(self.rel_logits.data.cpu().numpy(), axis=1).tolist()
        return predictions, probs

    def get_labels(self):
        # lazy transform to list
        self.labels = self.labels.cpu().numpy().tolist()
        return self.labels


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
