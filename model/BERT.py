import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from utils import constant

from model.bert import BertModel
from model.GCN import GCN
from model.DGA import DGAModel


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LabelAttention(nn.Module):
    def __init__(self, K_dim, V_dim, input_Q_dim, input_K_dim):
        super(LabelAttention, self).__init__()
        self.d_k = K_dim
        self.d_v = V_dim
        self.W_Q = nn.ModuleList()
        self.W_K = nn.ModuleList()
        for i in range(len(constant.LABEL_TO_ID) - 1):
            self.W_Q.append(nn.Linear(input_Q_dim, K_dim))
            self.W_K.append(nn.Linear(input_K_dim, K_dim))
        self.W_V = nn.Linear(input_Q_dim, self.d_v)
        self.W_out = nn.Linear(self.d_v, input_Q_dim)
        self.layerNorm = nn.LayerNorm(input_Q_dim)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        scores_sum = 0
        bs = Q.size()[0]
        K = K.unsqueeze(0).repeat(bs, 1, 1)  # [B x NL x LE]
        for i in range(len(constant.LABEL_TO_ID) - 1):
            Qi = self.W_Q[i](Q)
            Ki = self.W_K[i](K[:, i, :]).unsqueeze(1)
            scores = torch.matmul(Qi, Ki.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : # [B x L x 1]
            scores = scores + attn_mask.float() * -1e9
            scores = scores.transpose(-1, -2)  # [B x 1 x L]
            scores_sum = scores_sum + scores
        attn = nn.Softmax(dim=-1)(scores_sum)
        WV = self.W_V(V)
        context = torch.matmul(attn, WV).squeeze(1)  # [B x E]
        context = gelu(self.W_out(context))
        context = self.layerNorm(context)
        return context, attn


class EntityAttention(nn.Module):
    def __init__(self, K_dim, V_dim, input_Q_dim, input_K_dim):
        super(EntityAttention, self).__init__()
        self.d_k = K_dim
        self.d_v = V_dim
        self.W_Q = nn.Linear(input_Q_dim, K_dim)
        self.W_K = nn.Linear(input_K_dim, K_dim)

        self.W_V = nn.Linear(input_Q_dim, V_dim)
        self.W_out = nn.Linear(V_dim, input_Q_dim)
        self.layerNorm = nn.LayerNorm(input_Q_dim)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [B x entity Emb]

        Q = self.W_Q(Q)
        K = self.W_K(K).unsqueeze(1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : # [B x L x 1]
        scores = scores + attn_mask.float() * -1e9
        scores = scores.transpose(-1, -2)  # [B x 1 x L]
        attn = nn.Softmax(dim=-1)(scores)

        WV = self.W_V(V)
        context = torch.matmul(attn, WV).squeeze(1)  # [B x E]
        context = gelu(self.W_out(context))
        context = self.layerNorm(context)
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
        self.out_mlp = nn.Linear(config.hidden_size * 3, config.hidden_size)
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
        self.context_decoder = LabelAttention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.entity_mlp = nn.Linear(config.hidden_size * 2, opt["entity_hidden_dim"])

        hidden_size = opt["entity_hidden_dim"] + opt["V_dim"] + opt["gcn_hidden_dim"]
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
        label_indicator = torch.mean(label_embs, 0).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)  # [B x 1 x label_dim]
        attn_mask = subj_pos.eq(pos_indicator) + obj_pos.eq(pos_indicator)
        attn_mask = attn_mask.unsqueeze(2)  # [B x L x 1]
        context_feature, attn = self.context_decoder(inputs, label_indicator, inputs, attn_mask)  # [B x V_dim]

        # structure
        adj_gcn = adj
        if self.opt["deprel_edge"]:
            adj_gcn = adj.eq(0).eq(0)
        gcn_inputs = torch.cat([inputs, pos_embs, dep_embs], -1)
        gcn_outputs = self.structure_decoder(adj_gcn, gcn_inputs)  # [B x L x GCN_hidden]
        structure_feature = pool(gcn_outputs, pool_mask, pool_type)  # [B x GCN_hidden]

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


class Decoder3(nn.Module):
    def __init__(self, opt, config):
        super(Decoder3, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)

        self.context_decoder = LabelAttention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.feature_mlp = nn.Linear(config.hidden_size * 3, opt["hidden_dim"])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(opt["hidden_dim"], opt["label_dim"])
        # self.classifier = nn.Linear(opt["label_dim"], len(constant.LABEL_TO_ID))

    def forward(self, inputs, pos_ids, dep_ids, subj_pos, obj_pos, adj):
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask

        labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        pos_embs, dep_embs, label_embs = self.embs(labels, pos_ids, dep_ids)

        # context
        attn_mask = subj_pos.eq(pos_indicator) + obj_pos.eq(pos_indicator)
        attn_mask = attn_mask.unsqueeze(2)  # [B x L x 1]
        pad_mask = pos_ids.eq(0).unsqueeze(2)
        attn_mask = attn_mask + pad_mask
        context_feature, attn = self.context_decoder(inputs, label_embs, inputs, attn_mask)  # [B x V_dim]

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        obj_out = pool(inputs, obj_mask, type=pool_type)

        features = torch.cat([subj_out, obj_out, context_feature], -1)
        features = self.dropout(features)
        features = gelu(self.feature_mlp(features))
        features = gelu(self.out_mlp(features))

        labels = torch.tensor(list(range(0, 42)), dtype=torch.long).cuda()
        logits = self.scorer(self.embs.label_emb(labels), features)
        return logits, context_feature

    def scorer(self, label_embs, features):
        # label_embs : [N X E]
        # features : [B X E]
        batch_size = features.size()[0]
        num_labels = label_embs.size()[0]
        label_embs = label_embs.unsqueeze(0).repeat([batch_size, 1, 1])  # [B x N x E]
        features = features.unsqueeze(1).repeat([1, num_labels, 1])  # [B x N x E]
        scores = torch.sum(label_embs * features, -1)  # [B X N]
        # scores = torch.nn.functional.softmax(scores, -1)
        return scores


class Decoder4(nn.Module):
    """
    DGA => Entity Attention
    """
    def __init__(self, opt, config):
        super(Decoder4, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)
        self.structure_decoder = DGAModel(opt["dga_layers"], config.hidden_size,
                                          opt["K_dim"],
                                          opt["attn_hidden_dim"],
                                          opt["V_dim"],
                                          opt["num_heads"],
                                          opt["input_dropout"])
        self.entity_feature_mlp = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.structure_feature_extractor = EntityAttention(opt["K_dim"], opt["V_dim"],
                                                           config.hidden_size, config.hidden_size)

        # self.context_decoder = Attention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.feature_mlp = nn.Linear(config.hidden_size * 3, opt["hidden_dim"])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(opt["hidden_dim"], opt["label_dim"])
        self.classifier = nn.Linear(opt["label_dim"], len(constant.LABEL_TO_ID))

    def forward(self, inputs, pos_ids, dep_ids, subj_pos, obj_pos, adj):
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask

        attn_mask = subj_pos.eq(pos_indicator) + obj_pos.eq(pos_indicator)
        attn_mask = attn_mask.unsqueeze(2)  # [B x L x 1]
        pad_mask = pos_ids.eq(0).unsqueeze(2)
        attn_mask = attn_mask + pad_mask

        #labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        #pos_embs, dep_embs, label_embs = self.embs(labels, pos_ids, dep_ids)

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        obj_out = pool(inputs, obj_mask, type=pool_type)

        # structure
        dep_mask = get_mask_from_adj(adj)
        entity_feature = self.entity_feature_mlp(torch.cat([subj_out, obj_out], -1))
        structure_feature, _ = self.structure_decoder(inputs, dep_mask)
        structure_feature, attn = self.structure_feature_extractor(inputs, entity_feature, inputs, attn_mask)

        features = torch.cat([subj_out, obj_out, structure_feature], -1)
        features = self.dropout(features)
        features = gelu(self.feature_mlp(features))
        features = gelu(self.out_mlp(features))

        #labels = torch.tensor(list(range(0, 42)), dtype=torch.long).cuda()
        logits = self.classifier(features)
        return logits, structure_feature

    def scorer(self, label_embs, features):
        # label_embs : [N X E]
        # features : [B X E]
        batch_size = features.size()[0]
        num_labels = label_embs.size()[0]
        label_embs = label_embs.unsqueeze(0).repeat([batch_size, 1, 1])  # [B x N x E]
        features = features.unsqueeze(1).repeat([1, num_labels, 1])  # [B x N x E]
        scores = torch.sum(label_embs * features, -1)  # [B X N]
        # scores = torch.nn.functional.softmax(scores, -1)
        return scores


class Decoder5(nn.Module):
    """
    DGA => Label Attention
    """
    def __init__(self, opt, config):
        super(Decoder5, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)
        self.structure_decoder = DGAModel(opt["dga_layers"], config.hidden_size,
                                          opt["K_dim"],
                                          opt["attn_hidden_dim"],
                                          opt["V_dim"],
                                          opt["num_heads"],
                                          opt["input_dropout"])
        self.structure_feature_extractor = LabelAttention(opt["K_dim"], opt["V_dim"],
                                                           config.hidden_size, opt["label_dim"])

        # self.context_decoder = Attention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.feature_mlp = nn.Linear(config.hidden_size * 3, opt["hidden_dim"])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(opt["hidden_dim"], opt["label_dim"])

    def forward(self, inputs, pos_ids, dep_ids, subj_pos, obj_pos, adj):
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask

        attn_mask = subj_pos.eq(pos_indicator) + obj_pos.eq(pos_indicator)
        attn_mask = attn_mask.unsqueeze(2)  # [B x L x 1]
        pad_mask = pos_ids.eq(0).unsqueeze(2)
        attn_mask = attn_mask + pad_mask

        labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        pos_embs, dep_embs, label_embs = self.embs(labels, pos_ids, dep_ids)

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        obj_out = pool(inputs, obj_mask, type=pool_type)

        # structure
        dep_mask = get_mask_from_adj(adj)
        structure_feature, _ = self.structure_decoder(inputs, dep_mask)
        structure_feature, attn = self.structure_feature_extractor(inputs, label_embs, inputs, attn_mask)

        features = torch.cat([subj_out, obj_out, structure_feature], -1)
        features = self.dropout(features)
        features = gelu(self.feature_mlp(features))
        features = gelu(self.out_mlp(features))

        labels = torch.tensor(list(range(0, 42)), dtype=torch.long).cuda()
        logits = self.scorer(self.embs.label_emb(labels), features)
        return logits, structure_feature

    def scorer(self, label_embs, features):
        # label_embs : [N X E]
        # features : [B X E]
        batch_size = features.size()[0]
        num_labels = label_embs.size()[0]
        label_embs = label_embs.unsqueeze(0).repeat([batch_size, 1, 1])  # [B x N x E]
        features = features.unsqueeze(1).repeat([1, num_labels, 1])  # [B x N x E]
        scores = torch.sum(label_embs * features, -1)  # [B X N]
        # scores = torch.nn.functional.softmax(scores, -1)
        return scores

class BertRE(nn.Module):
    def __init__(self, opt, config):
        super(BertRE, self).__init__()
        self.opt = opt

        self.Encoder = Encoder(opt, config)
        # self.Decoder = Decoder2(opt, config)
        self.Decoder = Decoder3(opt, config)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = None
        self.rel_logits = None

    def forward(self, inputs):
        input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, subj_poses, obj_poses, bg_list, ed_list, subj_type_ids, obj_type_ids = inputs  # unpack

        self.labels = label_ids

        # Encoder
        seq_outputs = self.Encoder(input_ids, input_masks, segment_ids)[0]

        self.rel_logits, hidden_features = self.Decoder(seq_outputs, pos_ids, deprel_ids, subj_poses, obj_poses, adjs)
        if self.opt["match_loss_weight"] <= 0:
            loss = self.cal_rel_loss(self.rel_logits, label_ids)
        else:
            loss = self.cal_match_loss(self.rel_logits, label_ids)
        return loss, hidden_features

    def cal_rel_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def cal_match_loss(self, logits, labels):
        # logits : [B x N]
        # labels : [B]
        # get label scores
        batch_size = logits.size(0)
        class_num = len(constant.LABEL_TO_ID)
        one_hot = torch.zeros(batch_size, class_num).scatter_(1, labels.unsqueeze(1).cpu(), 1).cuda()
        label_scores = torch.max(logits + (one_hot.eq(0).float()*-9999999.0), -1)[0]
        # 扩大到[B X N]
        label_scores = label_scores.unsqueeze(1).repeat(1, class_num)  # [B x N]
        hinge_loss = logits - label_scores + 1
        # label 的位置设为0
        hinge_loss = hinge_loss * (one_hot.eq(0).float())
        hinge_loss = torch.clamp(hinge_loss, min=0)
        hinge_loss = torch.mean(hinge_loss, -1)
        hinge_loss = torch.sum(hinge_loss)
        return hinge_loss

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


def get_mask_from_adj(adj):

    mask = adj.eq(0)
    return mask.cuda()


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