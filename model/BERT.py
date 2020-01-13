import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from utils import constant

from model.bert import BertModel
from model.GCN import GCN
from model.DGA import DGAModel, UnitDGAModel
from model.BiDGA import BiDGAModel


def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LabelAttention(nn.Module):
    def __init__(self, K_dim, V_dim, input_Q_dim, input_K_dim, output_dim):
        super(LabelAttention, self).__init__()
        self.d_k = K_dim
        self.d_v = V_dim
        self.W_Q = nn.ModuleList()
        self.W_K = nn.ModuleList()
        for i in range(len(constant.LABEL_TO_ID) - 1):
            self.W_Q.append(nn.Linear(input_Q_dim, K_dim))
            self.W_K.append(nn.Linear(input_K_dim, K_dim))
        self.W_V = nn.Linear(input_Q_dim, output_dim)
        self.W_out = nn.Linear(self.d_v, output_dim)
        self.layerNorm = nn.LayerNorm(output_dim)

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
        attn = nn.Softmax(dim=-1)(scores_sum) # [B X 1 X L]

        # ======= attn 之后 原始 ======================
        WV = V
        context = torch.matmul(attn, WV).squeeze(1)  # [B x E]

        return context, attn


class LabelAttention2(nn.Module):
    def __init__(self, K_dim, V_dim, input_Q_dim, input_K_dim, output_dim):
        super(LabelAttention2, self).__init__()
        self.d_k = K_dim
        self.d_v = V_dim
        self.W_Q = nn.Linear(input_Q_dim, 256)
        self.W_K = nn.Linear(input_K_dim, 256)

    def forward(self, Q, K, V, attn_mask):
        # Q: [B x L x hidden size]
        # K: [Num Label-1 x Label Emb]
        # attn_mask : [B X SL X 1]
        bs = Q.size()[0]
        seq_len = Q.size()[1]
        num_label = K.size()[0]

        transfomed_K = self.W_K(K)
        transfomed_K = transfomed_K.unsqueeze(0).repeat(bs, 1, 1)  # [B x NL x E]
        K_expand = transfomed_K.unsqueeze(1).repeat(1, seq_len, 1, 1) # [B x SL X NL x E]

        transformed_Q = self.W_Q(Q) # [B x SL x E]
        Q_expand = transformed_Q.unsqueeze(2).repeat(1, 1, num_label, 1) / np.sqrt(self.d_k) # [B x SL X NL x E]

        S = torch.sum(K_expand * Q_expand, -1) # [B x SL X NL]

        scores = torch.sum(S, -1).unsqueeze(-1) + attn_mask.float() * -1e9 # [B x SL x 1]
        scores = scores.transpose(-1, -2) # [B x 1 X SL]
        attn = nn.Softmax(dim=-1)(scores) # [B x 1 X SL]

        WV = V
        context = torch.matmul(attn, WV).squeeze(1)  # [B x E]

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

    def forward(self, label_ids, pos_ids):
        pos_embs = self.pos_emb(pos_ids)
        label_embs = self.label_emb(label_ids)
        return pos_embs, label_embs


class Decoder4(nn.Module):
    """
    DGA 接(label+entity) attention
    entity 信息来自DGA
    """
    def __init__(self, opt, config):
        super(Decoder4, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)
        self.structure_decoder = DGAModel(num_layers=opt["dga_layers"],
                                          input_dim=config.hidden_size + opt["pos_dim"],
                                          attn_dim=opt["K_dim"],
                                          hidden_dim=opt["attn_hidden_dim"],
                                          v_dim=opt["V_dim"],
                                          feedforward_dim=opt["feedforward_dim"],
                                          num_heads=opt["num_heads"],
                                          dropout_prob=opt["input_dropout"])

        self.context_decoder = LabelAttention(K_dim=opt["K_dim"],
                                              V_dim=opt["V_dim"],
                                              input_Q_dim=opt["attn_hidden_dim"],
                                              input_K_dim=opt["label_dim"],
                                              output_dim=config.hidden_size)

        # self.context_decoder = Attention(opt["K_dim"], opt["V_dim"], config.hidden_size, opt["label_dim"])
        self.feature_mlp = nn.Linear(config.hidden_size * 3, opt["hidden_dim"])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_mlp = nn.Linear(opt["hidden_dim"], opt["label_dim"])

    def forward(self, inputs, pos_ids, subj_pos, obj_pos, adj, whole_adj):
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask


        # emb
        labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        pos_embs, label_embs = self.embs(labels, pos_ids)

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        obj_out = pool(inputs, obj_mask, type=pool_type)

        # structure
        dep_mask = get_mask_from_adj(whole_adj)
        dga_inputs = torch.cat([inputs, pos_embs], -1)
        structure_feature, _ = self.structure_decoder(dga_inputs, dep_mask)

        # context
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        context_feature, attn = self.context_decoder(structure_feature,
                                                     label_embs,
                                                     structure_feature, pool_mask)

        features = torch.cat([subj_out, obj_out, context_feature], -1)
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


class Decoder8(nn.Module):
    """
    Unit DGA(seq concat dep)

    """
    def __init__(self, opt, config):
        super(Decoder8, self).__init__()
        self.opt = opt
        self.config = config
        self.embs = EmbeddingLayer(opt)
        self.position_emb = nn.Embedding(500, opt["position_dim"])
        self.dist_emb = nn.Embedding(50, opt["dist_dim"])
        # self.structure_decoder = DGAModel(num_layers=opt["dga_layers"],
        #                                       seq_input_dim=config.hidden_size + 2*opt["position_dim"] + opt["pos_dim"],
        #                                       dep_input_dim=config.hidden_size + opt["dist_dim"] + opt["pos_dim"],
        #                                       attn_dim=opt["K_dim"],
        #                                       num_heads=opt["num_heads"],
        #                                       hidden_dim=opt["dga_hidden_dim"],
        #                                       feedforward_dim=opt["feedforward_dim"],
        #                                       dropout_prob=opt["input_dropout"])

        self.structure_decoder = BiDGAModel(num_layers=opt["dga_layers"],
                                            input_dim=config.hidden_size + opt["pos_dim"]+opt["dist_dim"],
                                            attn_dim=opt["K_dim"],
                                            v_dim=opt["V_dim"],
                                            hidden_dim=opt["hidden_dim"],
                                            feedforward_dim=opt["feedforward_dim"],
                                            dropout_prob=opt["input_dropout"])

        self.context_decoder = LabelAttention2(
            K_dim=opt["K_dim"], V_dim=opt["V_dim"],
            input_Q_dim=opt["hidden_dim"], input_K_dim=opt["label_dim"], output_dim=config.hidden_size)

        self.subj_mlp = nn.Linear(config.hidden_size, opt["hidden_dim"])
        self.obj_mlp = nn.Linear(config.hidden_size, opt["hidden_dim"])
        self.feature_mlp = nn.Linear(opt["hidden_dim"]*3, opt["hidden_dim"])

        self.dropout = nn.Dropout(opt["input_dropout"])
        self.out_mlp = nn.Linear(opt["hidden_dim"], opt["label_dim"])

    def forward(self, inputs, pos_ids, subj_pos, obj_pos, dist, adj, whole_adj, ancestor_adj, input_mask):
        pos_indicator = int(self.opt["max_seq_length"])
        pool_type = self.opt['pooling']
        seq_len = inputs.size()[1]
        subj_mask = subj_pos.eq(pos_indicator).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(pos_indicator).eq(0).unsqueeze(2)  # invert mask

        # emb
        labels = torch.tensor(list(range(1, 42)), dtype=torch.long).cuda()
        pos_embs, label_embs = self.embs(labels, pos_ids)
        subj_postion_embs = self.position_emb(subj_pos)
        obj_position_embs = self.position_emb(obj_pos)
        dist_embs = self.dist_emb(dist)

        # Entity
        subj_out = pool(inputs, subj_mask, type=pool_type)
        subj_feature = self.subj_mlp(subj_out)
        obj_out = pool(inputs, obj_mask, type=pool_type)
        obj_feature = self.obj_mlp(obj_out)

        # structure
        forward_dep_mask = get_mask_from_adj(whole_adj)
        backward_dep_mask = get_mask_from_adj(ancestor_adj)
        dep_dga_inputs = torch.cat([inputs, pos_embs, dist_embs], -1)
        # seq_dga_inputs = torch.cat([inputs, pos_embs, subj_postion_embs, obj_position_embs], -1)

        l = input_mask.cpu().numpy().astype(np.int64).sum(1)
        _, seq_mask = get_attn_masks(torch.Tensor(l).long().cuda(), seq_len)
        pad_mask = input_mask.eq(0).unsqueeze(1).repeat(1, seq_len, 1)
        seq_mask = seq_mask + pad_mask

        structure_feature, _ = self.structure_decoder(dep_dga_inputs, dep_dga_inputs,
                                                      forward_dep_mask, backward_dep_mask)

        # context
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        # ==== 原始 ======
        # pool_mask = pos_ids.eq(0).unsqueeze(2)
        context_feature, attn = self.context_decoder(structure_feature,
                                                     label_embs,
                                                     structure_feature, pool_mask)

        features = torch.cat([subj_feature, obj_feature, context_feature], -1)
        features = self.dropout(features)
        features = gelu(self.feature_mlp(features))
        features = self.out_mlp(features)

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
        self.Decoder = Decoder8(opt, config)

        weight = np.ones([len(constant.LABEL_TO_ID)], dtype=np.float32)
        weight[0] = opt["neg_weight"]
        self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float().cuda())
        # self.criterion = nn.CrossEntropyLoss()
        self.labels = None
        self.rel_logits = None

    def forward(self, inputs):
        input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, subj_poses, obj_poses, bg_list, ed_list, \
        subj_type_ids, obj_type_ids, whole_adjs, ancestor_adjs = inputs  # unpack

        self.labels = label_ids

        # Encoder
        seq_outputs = self.Encoder(input_ids, input_masks, segment_ids)[0]

        # self.rel_logits, hidden_features = self.Decoder(seq_outputs, pos_ids, subj_poses, obj_poses,
        #                                                 dists, adjs, whole_adjs, ancestor_adjs, input_masks)
        self.rel_logits, hidden_features = self.Decoder(seq_outputs, pos_ids, subj_poses, obj_poses,
                                                        dists, adjs, whole_adjs, ancestor_adjs, input_masks)
        if self.opt["match_loss_weight"] <= 0:
            loss = self.cal_rel_loss(self.rel_logits, label_ids)
        else:
            loss = self.cal_match_loss(self.rel_logits, label_ids)
        return loss, hidden_features

    def cal_rel_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def cal_match_loss(self, logits, labels, M: int = 1):
        # logits : [B x N]
        # labels : [B]
        # get label scores
        batch_size = logits.size(0)
        class_num = len(constant.LABEL_TO_ID)
        one_hot = torch.zeros(batch_size, class_num).scatter_(1, labels.unsqueeze(1).cpu(), 1).cuda()
        label_scores = torch.max(logits + (one_hot.eq(0).float() * -9999999.0), -1)[0]
        # 扩大到[B X N]
        label_scores = label_scores.unsqueeze(1).repeat(1, class_num)  # [B x N]
        hinge_loss = logits - label_scores + M
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