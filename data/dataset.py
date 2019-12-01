"""
一些注意事项
1. 矩阵必须在这生成，因为head subj_pos 等 可能会被截断，
所以先生成完整矩阵，如果需要截断，直接取矩阵的子矩阵即可，
防止生产邻接矩阵时找不到头节点


2. 两种处理subword的方式,通过 opt["subword_to_children"] 设置: True为处理方式(1) False 为处理方式(2)
    (1) subword 当作子节点, dist 的计算 和正常的一样
    (2) subword 等同原节点，子节点将会有多个父节点,
        计算LCA dist的思路：用原来的老head生成一波dist，subword的dist等于这个dist中原来单词的位置的值

3. Adj 最后的长度和bert sequence 长度一样，已经在最前面和最后面 PAD了0 ，可直接用于GCN 或 Dependency Guided Attention

4. Ner 对Subword 有两种处理方式, 通过 opt["first_subword_ner"] 设置
    (1) 所有的subword 共享同一个ner
    (2) 只有第一个subword 使用ner, 后面的subword都是pad, 可用于联合抽取 或 多任务

5. subword mask : 每个单词的第一个subword位置为1 ，[CLS] [SEP] [PAD] 其余subword 都为0

6. opt["entity_mask"] ： 如果为True 实体用NER 代替， 而不用真实单词；否则使用原单词

7. opt["only_child"]: 如果为True, 生成矩阵时，只连接结点和它的孩子，不连接父节点；否则 父节点和子节点都相连

8. opt["self_loop"]: 如果为True, 生成矩阵的对角线将有值： 1 或 self deprel id,否则对角线为 0

9. opt["deprel_edge"]: 如果为 True, 生成的邻接矩阵的边为deprel id， 否则为1

"""

import os
import numpy as np
import json
import pickle
import re
import torch

from torch.utils.data import Dataset

from utils import constant, helper, golVars
from model.tree import head_to_tree, tree_to_adj


import logging
logger = logging.getLogger(__name__)
output_examples = True


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, opt, single_data):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = single_data["id"]
        self.words = single_data["token"]
        self.label = single_data["relation"]
        self.pos = single_data['stanford_pos']
        self.ner = single_data['stanford_ner']
        self.head = [int(x) for x in single_data['stanford_head']]
        self.deprel = single_data['stanford_deprel']
        self.subj_type = single_data['subj_type']
        self.subj_start = single_data['subj_start']
        self.subj_end = single_data['subj_end']
        self.obj_type = single_data['obj_type']
        self.obj_start = single_data['obj_start']
        self.obj_end = single_data['obj_end']


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, length, subword_mask, label_id,
                 ner_ids, pos_ids, deprel_ids, subj_pos, obj_pos,
                 bg_list, ed_list, subj_type, obj_type, heads, old_heads):
        self.tokens = tokens
        self.length = length
        self.label_id = label_id
        self.ner_ids = ner_ids
        self.pos_ids = pos_ids
        self.deprel_ids = deprel_ids
        self.bg_list = bg_list
        self.ed_list = ed_list
        self.subword_mask = subword_mask
        self.subj_type = subj_type
        self.obj_type = obj_type
        self.subj_pos = subj_pos
        self.obj_pos = obj_pos
        self.old_heads = old_heads
        self.heads = heads


def read_examples_from_file(opt, data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        all_data = json.load(f)
        for single_data in all_data:
            ss = single_data["subj_start"]
            se = single_data["subj_end"]
            st = single_data["subj_type"]
            objs = single_data["obj_start"]
            obje = single_data["obj_end"]
            ot = single_data["obj_type"]
            if opt["entity_mask"]:
                single_data["token"][ss:se + 1] = ['SUBJ-' + st] * (se - ss + 1)
                single_data["token"][objs:obje + 1] = ['OBJ-' + ot] * (obje - objs + 1)

            for i, t in enumerate(single_data["token"]):
                # 实现Entity Mask
                if "LRB" in t:
                    single_data["token"][i] = '“'
                elif "RRB" in t:
                    single_data["token"][i] = '”'
                elif t in constant.ADDITIONAL_WORDS:
                    continue
                elif t == "co-founder":
                    single_data["token"][i] = "founder"
                elif re.match(r'[=+*-.#$@!~_—]+$', t) is not None:
                    single_data["token"][i] = t[0]

                if len(t) > 5:
                    if "http:" in t or "https:" in t or "www" in t or ".org/" in t or \
                            (".com" in t and "@" not in t) or (".ss" in t and "@" not in t):
                        single_data["token"][i] = "WEB-URL"
                    elif ("com" in t and "@" in t) or (".net" in t and "@" in t) or (".ss" in t and "@" in t):
                        single_data["token"][i] = "E-MAIL"
                    elif re.match(r'(([A-Z]*[0-9][A-Z]*)+[-]([A-Z]*[0-9][A-Z]*)+)+', t) is not None:
                        single_data["token"][i] = "E-CODE"
                    elif re.match(r'(,*[0-9])+', t) is not None or re.match(r'([A-Za-z]*[0-9][A-Za-z]*){4,}', t) is not None:
                        single_data["token"][i] = "E-NUM"
                    else:
                        continue

            examples.append(InputExample(opt, single_data))
    return examples


def convert_examples_to_features(opt,
                                 examples,
                                 tokenizer,
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        label_id = constant.LABEL_TO_ID[example.label]
        tokens = []
        pos_ids = []
        ner_ids = []
        heads = []
        deprel_ids = []
        old_index2new_index = []
        old_end2new_end = []
        idx = 0
        word_tokens_tmp = [] # 备份word_wokens 用于新的head的计算
        subword_mask = []

        # 如果分词后，长度大于128 就不分词了
        canTokenize = testCanTokenize(example.words, tokenizer)

        for word, pos, ner, deprel in zip(example.words, example.pos, example.ner, example.deprel):
            # 更新old2new的映射
            old_index2new_index.append(len(tokens))
            # 获取分词
            word_tokens = tokenizeWord(word, tokenizer, canTokenize)
            word_tokens_tmp.append(word_tokens)
            tokens.extend(word_tokens)
            # 更新old_end2new_end的映射
            old_end2new_end.append(len(tokens) - 1)

            # 被拆分的保存同样的pos
            pos_ids.extend([constant.POS_TO_ID[pos]] * len(word_tokens))

            # 获取ner
            if opt["first_subword_ner"]:
                # 只有第一个subword 有ner
                ner_ids.extend([constant.NER_TO_ID[ner]] + [constant.PAD_ID] * (len(word_tokens) - 1))
            else:
                # 所有subword 保存同样的ner
                ner_ids.extend([constant.NER_TO_ID[ner]] * len(word_tokens))

            # 获取deprel id
            if opt["subword_to_children"]:
                # 如果单词被拆分，第一部分的头部为原来， 拆开的剩下的单词，以第一个单词为头
                deprel_ids.extend([constant.DEPREL_TO_ID[deprel]] + [constant.DEPREL_TO_ID[constant.SAME_TOKEN_DEP]] * (
                        len(word_tokens) - 1))
            else:
                # 如果单词被拆分， subword共享同样的依存边类型
                deprel_ids.extend([constant.DEPREL_TO_ID[deprel]]*len(word_tokens))

            # 获取 subword mask
            subword_mask.extend([1]+[0] * (len(word_tokens) - 1)) # 第一个subword 为1, 其余为0

            idx += 1
        max_len = max(len(tokens), max_len)
        # 获取 head
        if opt["subword_to_children"]:
            # 如果单词被拆分，第一部分的头部为原来， 拆开的剩下的单词，以第一个单词为头
            # 累加前面拆分的长度
            for i, h in enumerate(example.head):
                word_tokens = word_tokens_tmp[i]
                # head 是真实下标
                this_word_index = i + 1 # 作为head 要 + 1
                # 单独处理 head = 0的情况
                if h == 0:
                    new_h = 0
                else:
                    # h 是真实下标所以先 -1 索引到头节点新的下标, 为了还原真实下标，还要 +1
                    new_h = old_index2new_index[h-1] + 1
                heads.extend([new_h] + [this_word_index] * (len(word_tokens) - 1))
                ns = old_index2new_index[i]
                ne = old_end2new_end[i]
                assert len(word_tokens) == (ne-ns+1)
        else:
            # 将subword 等同看待，将会有多对多的连接，父节点可能有多个，自身可能有多个，子节点可能有多个
            # heads 的每一个元素是一个列表，包含老head的所有新的subword的索引
            for i, h in enumerate(example.head):
                word_tokens = word_tokens_tmp[i]
                ns = old_index2new_index[i]
                ne = old_end2new_end[i]
                assert len(word_tokens) == (ne-ns+1)
                if h == 0:
                    new_h_s = 0
                    new_h_e = 0
                else:
                    new_h_s = old_index2new_index[h-1] + 1
                    new_h_e = old_end2new_end[h-1] + 1
                new_h = list(range(new_h_s, new_h_e+1))
                heads.extend([new_h] * len(word_tokens))

        # 处理subj_position 和 obj_position
        l = len(tokens)
        subj_start = old_index2new_index[example.subj_start]
        subj_end = old_end2new_end[example.subj_end]
        obj_start = old_index2new_index[example.obj_start]
        obj_end = old_end2new_end[example.obj_end]
        subj_pos = get_positions(subj_start, subj_end, l)
        obj_pos = get_positions(obj_start, obj_end, l)

        seq_len = len(tokens)
        assert len(subword_mask) == len(tokens)
        assert len(ner_ids) == seq_len
        assert len(pos_ids) == seq_len
        assert len(subj_pos) == seq_len
        assert len(obj_pos) == seq_len
        assert len(heads) == seq_len
        assert len(deprel_ids) == seq_len
        assert len(old_end2new_end) == len(example.head)

        features.append(InputFeatures(tokens, seq_len, subword_mask, label_id, ner_ids, pos_ids, deprel_ids,
                                      subj_pos, obj_pos, old_index2new_index, old_end2new_end,
                                      example.subj_type, example.obj_type, heads, example.head))
    print(" MAX LEN IS : {}".format(max_len))
    return features


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, maxlen, deprels,
                        only_child=False, self_loop=True, deprel_edge=False):
    tree, dist = head_to_tree(head, words, l, prune, subj_pos, obj_pos)
    # adj 邻接边为边类型
    adj = tree_to_adj(maxlen, l, tree, only_child, self_loop)

    if deprel_edge:
        for i, h in enumerate(head):
            # 如果被剪枝了就跳过
            if adj[h-1][i] == 0:
                continue
            dep = deprels[i]
            adj[h-1, i] = dep # 边的值设为deprel id
            if not only_child:
                # 如果结点也连接了父节点，与父节点的边类型为 r-deprel id
                adj[i, h-1] = dep + constant.DEPREL_COUNT
            if self_loop:
                # 如果自环，则把边类型设为自环的deprel id
                adj[i, i] = constant.DEPREL_TO_ID[constant.SELF_DEP]

    return adj, dist


def heads_to_adj(heads, deprels,  maxlen, old_heads, subj_pos, obj_pos,
                 bg_list, ed_list, only_child=False, self_loop=True, deprel_edge=False):
    """ 每一个节点可能有多个head,可能多对多 """
    ret = np.zeros((maxlen, maxlen), dtype=np.float32)
    for i in range(len(heads)):
        hs = heads[i]
        dep = deprels[i]
        for h in hs:
            if h == 0:
                continue
            ret[h-1, i] = 1 if not deprel_edge else dep
            if not only_child:
                ret[i, h-1] = 1 if not deprel_edge else dep+constant.DEPREL_COUNT

    if self_loop:
        for i in range(len(heads)):
            ret[i, i] = 1 if not deprel_edge else constant.DEPREL_TO_ID[constant.SELF_DEP]

    # 计算dists
    _, old_dist = head_to_tree(old_heads, [""], len(old_heads), -1, subj_pos, obj_pos)
    assert len(old_dist) == len(old_heads)
    new_dist = []
    for i, d in enumerate(old_dist):
        ns = bg_list[i]
        ne = ed_list[i]
        word_lenth = ns-ne+1
        new_dist.extend([d]*word_lenth) # 所有subword 共享同样的 LCA distance

    assert len(heads) == len(new_dist)

    return ret, new_dist


class RelationDataset(Dataset):
    def __init__(self, opt, mode, tokenizer):
        self.opt = opt
        self.tokenizer = tokenizer
        self.output_examples = False
        cached_features_file = os.path.join(opt["data_dir"], "cached_{}.data".format(mode))
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as f:
                self.features = pickle.load(f)
        else:
            logger.info("Creating features from dataset file at %s", opt["data_dir"])
            examples = read_examples_from_file(opt, opt["data_dir"], mode)
            self.features = convert_examples_to_features(self.opt, examples, self.tokenizer)
            with open(cached_features_file, 'wb') as f:
                pickle.dump(self.features, f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def collate_fn(features):
    opt = golVars.get_value("OPT")
    tokenizer = golVars.get_value("TKZ")
    batch_data = convertData(opt, features, tokenizer,
                             cls_token=tokenizer.cls_token,
                             sep_token=tokenizer.sep_token,
                             pad_token=0)
    golVars.set_value("OUTPUT_EXAMPLES", False)
    return batch_data


def convertData(opt, features, tokenizer,
                cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=0,
                sep_token="[SEP]", sep_token_extra=False,
                pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                sequence_a_segment_id=0, mask_padding_with_zero=True):

    lengths = [f.length for f in features]
    batch_max_len = max(lengths)
    feature_cnt = 0

    input_ids_lst, input_masks_lst, subword_masks_lst, segment_ids_lst, label_ids, ner_ids_lst, pos_ids_lst,\
    deprel_ids_lst, adjs, dists_lst, bg_list_lst, ed_list_lst, subj_type_ids, obj_type_ids, \
    subj_pos_lst, obj_pos_lst = [list() for _ in range(16)]

    for feature in features:
        # 生成邻接矩阵,
        # max(l, max_seq_length-2) ： 如果 l> max_seq_length-2保证矩阵是完整的， 如果l < max_seq_length-2, 保证矩阵长度max_seq_length-2
        # len(dists) == l 仍需要pad
        if opt["subword_to_children"]:
            adj, dists = inputs_to_tree_reps(feature.heads, feature.tokens, feature.length,
                                             opt['prune_k'], feature.subj_pos, feature.obj_pos,
                                             batch_max_len, feature.deprel_ids,
                                             opt["only_child"], opt["self_loop"], opt["deprel_edge"])
        else:
            adj, dists = heads_to_adj(feature.heads, feature.deprel_ids, batch_max_len,
                                      feature.old_heads,
                                      feature.subj_pos, feature.obj_pos,
                                      feature.bg_list, feature.ed_list,
                                      only_child=opt["only_child"], self_loop=opt["self_loop"],
                                      deprel_edge=opt["deprel_edge"])

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = feature.tokens + [sep_token]
        ner_ids = feature.ner_ids + [constant.PAD_ID]
        pos_ids = feature.pos_ids + [constant.PAD_ID]
        deprel_ids = feature.deprel_ids + [constant.PAD_ID]
        subj_pos = feature.subj_pos + [feature.subj_pos[-1] + 1]
        obj_pos = feature.obj_pos + [feature.obj_pos[-1] + 1]
        subword_mask = feature.subword_mask + [0]
        dists += [0]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            ner_ids += [constant.PAD_ID]
            pos_ids += [constant.PAD_ID]
            deprel_ids += [constant.PAD_ID]
            subj_pos += [subj_pos[-1] + 1]
            obj_pos += [obj_pos[-1] + 1]
            subword_mask += [0]

            dists += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            ner_ids = [constant.PAD_ID] + ner_ids
            pos_ids = [constant.PAD_ID] + pos_ids
            deprel_ids = [constant.PAD_ID] + deprel_ids
            subj_pos = [subj_pos[0] - 1] + subj_pos
            obj_pos = [obj_pos[0] - 1] + obj_pos
            subword_mask = [0] + subword_mask

            dists = [0] + dists
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = batch_max_len + 2 - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            dists = ([0] * padding_length) + dists

            ner_ids = ([constant.PAD_ID] * padding_length) + ner_ids
            pos_ids = ([constant.PAD_ID] * padding_length) + pos_ids
            deprel_ids = ([constant.PAD_ID] * padding_length) + deprel_ids
            subword_mask = ([0] * padding_length) + subword_mask

            for i in range(padding_length):
                subj_pos = [subj_pos[0] - 1] + subj_pos
                obj_pos = [obj_pos[0] - 1] + obj_pos

        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            dists += ([0] * padding_length)

            ner_ids += ([constant.PAD_ID] * padding_length)
            pos_ids += ([constant.PAD_ID] * padding_length)
            deprel_ids += ([constant.PAD_ID] * padding_length)
            subword_mask += ([0] * padding_length)
            for i in range(padding_length):
                subj_pos += [subj_pos[-1] + 1]
                obj_pos += [obj_pos[-1] + 1]

        # 单独处理 adj
        adj = np.pad(adj, ((1, 1), (1, 1)), 'constant', constant_values=(0.0, 0.0))

        subj_pos = [p + opt["max_seq_length"] for p in subj_pos]
        obj_pos = [p + opt["max_seq_length"] for p in obj_pos]

        assert len(input_ids) == batch_max_len + 2
        assert len(input_mask) == batch_max_len + 2
        assert len(segment_ids) == batch_max_len + 2
        assert len(dists) == batch_max_len + 2
        assert adj.shape[0] == batch_max_len + 2

        assert len(ner_ids) == batch_max_len + 2
        assert len(pos_ids) == batch_max_len + 2
        assert len(obj_pos) == batch_max_len + 2
        assert len(deprel_ids) == batch_max_len + 2
        assert len(subword_mask) == batch_max_len + 2
        assert len(subj_pos) == batch_max_len + 2
        assert len(obj_pos) == batch_max_len + 2

        if golVars.get_value("OUTPUT_EXAMPLES") and feature_cnt < 5:
            logger.info("*** Example ***")
            logger.info("Label Id: {}".format(feature.label_id))
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("subword_mask: %s", " ".join([str(x) for x in subword_mask]))
            logger.info("ner_ids: %s", " ".join([str(x) for x in ner_ids]))
            logger.info("subj_position: %s", " ".join([str(x) for x in subj_pos]))
            logger.info("LCA distances: %s", " ".join([str(x) for x in dists]))
            logger.info("Adj(front 10 node): {}".format(adj[:10, :10]))
            feature_cnt += 1

        label_ids.append(feature.label_id)
        subj_type_ids.append(constant.SUBJ_NER_TO_ID[feature.subj_type])
        obj_type_ids.append(constant.OBJ_NER_TO_ID[feature.obj_type])
        adjs.append(adj)
        bg_list_lst.append(feature.bg_list)
        ed_list_lst.append(feature.ed_list)

        input_ids_lst.append(input_ids)
        input_masks_lst.append(input_mask)
        segment_ids_lst.append(segment_ids)
        dists_lst.append(dists)
        subword_masks_lst.append(subword_mask)
        ner_ids_lst.append(ner_ids)
        pos_ids_lst.append(pos_ids)
        deprel_ids_lst.append(deprel_ids)
        subj_pos_lst.append(subj_pos)
        obj_pos_lst.append(obj_pos)

    input_ids_lst = torch.tensor(input_ids_lst, dtype=torch.long).cuda()
    dists_lst = torch.tensor(dists_lst, dtype=torch.long).cuda()
    segment_ids_lst = torch.tensor(segment_ids_lst, dtype=torch.long).cuda()

    input_masks_lst = torch.tensor(input_masks_lst, dtype=torch.long).cuda()
    # subword_masks_lst = torch.tensor(subword_masks_lst, dtype=torch.long).cuda()
    ner_ids_lst = torch.tensor(ner_ids_lst, dtype=torch.long).cuda()
    pos_ids_lst = torch.tensor(pos_ids_lst, dtype=torch.long).cuda()
    deprel_ids_lst = torch.tensor(deprel_ids_lst, dtype=torch.long).cuda()
    adjs = torch.tensor(adjs, dtype=torch.float).cuda()
    subj_pos_lst = torch.tensor(subj_pos_lst, dtype=torch.long).cuda()
    obj_pos_lst = torch.tensor(obj_pos_lst, dtype=torch.long).cuda()
    label_ids = torch.tensor(label_ids, dtype=torch.long).cuda()

    return input_ids_lst, input_masks_lst, subword_masks_lst, segment_ids_lst, label_ids, ner_ids_lst, pos_ids_lst, \
           deprel_ids_lst, adjs, dists_lst, subj_pos_lst, obj_pos_lst, bg_list_lst, ed_list_lst, subj_type_ids, obj_type_ids


def tokenizeWord(word, tokenizer, canTokenize=True):
    if word in constant.ADDITIONAL_WORDS or word in tokenizer.vocab:
        return [word]
    elif word[0] + word[1:] in tokenizer.vocab:
        return [word[0] + word[1:]]
    elif word.lower() in tokenizer.vocab:
        return [word.lower()]
    if canTokenize:
        word_tokens = tokenizer.tokenize(word)
    else:
        word_tokens = [word]
    # if 5 < len(word_tokens) <= 7:
    #     print(word)
    #     print(word_tokens)
    #     print("================================================")
    if len(word_tokens) >= 4:
        word_tokens = [word]
    return word_tokens


def testCanTokenize(words, tokenizer):
    tokens = []
    for word in words:
        # 更新old2new的映射
        word_tokens = tokenizeWord(word, tokenizer)
        tokens.extend(word_tokens)
    if len(tokens) > 126:
        # logger.info("Over 128 Length tokens: %s", " ".join([str(x) for x in tokens]))
        return False
    else:
        return True

