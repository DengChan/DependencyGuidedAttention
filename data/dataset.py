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
import torch
from utils import constant
from model.tree import head_to_tree, tree_to_adj
from torch.utils.data import TensorDataset, Dataset


import logging
logger = logging.getLogger(__name__)


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

    def __init__(self, input_ids, input_mask, subword_mask, segment_ids, label_id,
                 ner_ids, pos_ids, deprel_ids, adj, dists, subj_pos, obj_pos,
                 bg_list, ed_list, subj_type, obj_type):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ner_ids = ner_ids
        self.pos_ids = pos_ids
        self.deprel_ids = deprel_ids
        self.adj = adj
        self.dists = dists
        self.bg_list = bg_list
        self.ed_list = ed_list
        self.subword_mask = subword_mask
        self.subj_type = subj_type
        self.obj_type = obj_type
        self.subj_pos = subj_pos
        self.obj_pos = obj_pos


def read_examples_from_file(opt, data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        all_data = json.load(f)
        for single_data in all_data:
            for i, t in enumerate(single_data["token"]):
                if "LRB" in t:
                    single_data["token"][i] = '“'
                elif "RRB" in t:
                    single_data["token"][i] = '”'
            examples.append(InputExample(opt, single_data))
    return examples


def convert_examples_to_features(opt,
                                 examples,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
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
        for word, pos, ner, deprel in zip(example.words, example.pos, example.ner, example.deprel):
            # 更新old2new的映射
            old_index2new_index.append(len(tokens))
            # 获取分词
            word_tokens = tokenizer.tokenize(word)
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

        assert len(example.head) == len(word_tokens_tmp)

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
                    new_h_s = old_index2new_index[h-1]+1
                    new_h_e = old_end2new_end[h-1] + 1
                new_h = list(range(new_h_s, new_h_e+1))
                heads.extend([new_h]*len(word_tokens))

        # 处理subj_position 和 obj_position
        l = len(tokens)
        subj_start = old_index2new_index[example.subj_start]
        subj_end = old_end2new_end[example.subj_end]
        obj_start = old_index2new_index[example.obj_start]
        obj_end = old_end2new_end[example.obj_end]
        subj_pos = get_positions(subj_start, subj_end, l)
        obj_pos = get_positions(obj_start, obj_end, l)

        special_tokens_count = 3 if sep_token_extra else 2
        # 生成邻接矩阵,
        # max(l, max_seq_length-2) ： 如果 l> max_seq_length-2保证矩阵是完整的， 如果l < max_seq_length-2, 保证矩阵长度max_seq_length-2
        # len(dists) == l 仍需要pad
        if opt["subword_to_children"]:
            adj, dists = inputs_to_tree_reps(heads, tokens, l, opt['prune_k'], subj_pos, obj_pos,
                                             max(l, max_seq_length-special_tokens_count), deprel_ids,
                                             opt["only_child"], opt["self_loop"], opt["deprel_edge"])
        else:
            adj, dists = heads_to_adj(heads, deprel_ids, max(l, max_seq_length-special_tokens_count), example.head,
                                      subj_pos, obj_pos, old_index2new_index, old_end2new_end,
                                      only_child=opt["only_child"], self_loop=opt["self_loop"],
                                      deprel_edge=opt["deprel_edge"])

        # 实现Entity Mask
        if opt["entity_mask"]:
            tokens[subj_start:subj_end + 1] = ['SUBJ-' + example.subj_type] * (subj_end - subj_start + 1)
            tokens[obj_start:obj_end + 1] = ['OBJ-' + example.obj_type] * (obj_end - obj_start + 1)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # 截断大于最大长度的序列
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            pos_ids = pos_ids[:(max_seq_length - special_tokens_count)]
            ner_ids = ner_ids[:(max_seq_length - special_tokens_count)]
            deprel_ids = deprel_ids[:(max_seq_length - special_tokens_count)]
            subj_pos = subj_pos[:(max_seq_length - special_tokens_count)]
            obj_pos = obj_pos[:(max_seq_length - special_tokens_count)]
            dists = dists[:(max_seq_length - special_tokens_count)]
            subword_mask = subword_mask[:(max_seq_length - special_tokens_count)]

        # 截断大于最大长度的邻接矩阵元素
        if adj.shape[0] > max_seq_length-special_tokens_count:
            adj = adj[:max_seq_length-special_tokens_count, :max_seq_length-special_tokens_count]

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
        tokens += [sep_token]
        ner_ids += [constant.PAD_ID]
        pos_ids += [constant.PAD_ID]
        deprel_ids += [constant.PAD_ID]
        subj_pos += [subj_pos[-1] + 1]
        obj_pos += [obj_pos[-1] + 1]
        dists += [0]
        subword_mask += [0]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            pos_ids += [constant.PAD_ID]
            ner_ids += [constant.PAD_ID]
            deprel_ids += [constant.PAD_ID]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            ner_ids += [constant.PAD_ID]
            segment_ids += [cls_token_segment_id]
            pos_ids += [constant.PAD_ID]
            deprel_ids += [constant.PAD_ID]
            subj_pos += [subj_pos[0] - 1]
            obj_pos += [obj_pos[0] - 1]
            dists += [0]
            subword_mask += [0]
        else:
            tokens = [cls_token] + tokens
            ner_ids = [constant.PAD_ID] + ner_ids
            pos_ids = [constant.PAD_ID] + pos_ids
            deprel_ids = [constant.PAD_ID] + deprel_ids
            subj_pos = [subj_pos[0]-1] + subj_pos
            obj_pos = [obj_pos[0]-1] + obj_pos
            dists = [0] + dists
            segment_ids = [cls_token_segment_id] + segment_ids
            subword_mask = [0] + subword_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            ner_ids = ([constant.PAD_ID] * padding_length) + ner_ids
            pos_ids = ([constant.PAD_ID] * padding_length) + pos_ids
            deprel_ids = ([constant.PAD_ID] * padding_length) + deprel_ids
            dists = ([0] * padding_length) + dists
            subword_mask = ([0] * padding_length) + subword_mask
            for i in range(padding_length):
                subj_pos = [subj_pos[0] - 1] + subj_pos
                obj_pos = [obj_pos[0] - 1] + obj_pos

        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            ner_ids += ([constant.PAD_ID] * padding_length)
            pos_ids += ([constant.PAD_ID] * padding_length)
            deprel_ids += ([constant.PAD_ID] * padding_length)
            dists += ([0] * padding_length)
            subword_mask += ([0] * padding_length)
            for i in range(padding_length):
                subj_pos += [subj_pos[-1] + 1]
                obj_pos += [obj_pos[-1] + 1]

        # 单独处理 adj
        adj = np.pad(adj, ((1, 1), (1, 1)), 'constant', constant_values=(0.0, 0.0))

        subj_pos = [p + max_seq_length for p in subj_pos]
        obj_pos = [p + max_seq_length for p in obj_pos]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ner_ids) == max_seq_length
        assert len(subj_pos) == max_seq_length
        assert len(dists) == max_seq_length
        assert adj.shape[0] == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("Label Id: {}".format(label_id))
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("subword_mask: %s", " ".join([str(x) for x in subword_mask]))
            logger.info("ner_ids: %s", " ".join([str(x) for x in ner_ids]))
            logger.info("subj_position: %s", " ".join([str(x) for x in subj_pos]))
            logger.info("LCA distances: %s", " ".join([str(x) for x in dists]))
            logger.info("Adj(front 10 node): {}".format(adj[:10, :10]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              subword_mask=subword_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              ner_ids=ner_ids,
                              pos_ids=pos_ids,
                              deprel_ids=deprel_ids,
                              adj=adj,
                              dists=dists,
                              subj_pos=subj_pos,
                              obj_pos=obj_pos,
                              bg_list=old_index2new_index,
                              ed_list=old_end2new_end,
                              subj_type=constant.SUBJ_NER_TO_ID[example.subj_type],
                              obj_type=constant.OBJ_NER_TO_ID[example.obj_type]))
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
    def __init__(self, opt, mode, tokenizer, pad_token_ner_label_id):
        cached_features_file = os.path.join(opt["data_dir"], "cached_{}.data".format(mode))
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            self.features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", opt["data_dir"])
            examples = read_examples_from_file(opt, opt["data_dir"], mode)
            self.features = convert_examples_to_features(opt, examples, opt["max_seq_length"], tokenizer,
                                                         cls_token=tokenizer.cls_token,
                                                         sep_token=tokenizer.sep_token,
                                                         # pad on the left for xlnet
                                                         pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                         pad_token_label_id=pad_token_ner_label_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def collate_fn(data):

    input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, bg_list, ed_list, subj_type_ids, obj_type_ids, subj_poses, obj_poses = \
        [list() for _ in range(16)]
    # data.sort(key=lambda x: len(x.bg_list), reverse=True)
    for item in data:
        input_ids.append(item.input_ids)
        input_masks.append(item.input_mask)
        subword_masks.append(item.subword_mask)
        segment_ids.append(item.segment_ids)
        label_ids.append(item.label_id)
        ner_ids.append(item.ner_ids)
        pos_ids.append(item.pos_ids)
        deprel_ids.append(item.deprel_ids)
        adjs.append(item.adj)
        dists.append(item.dists)
        subj_poses.append(item.subj_pos)
        obj_poses.append(item.obj_pos)
        bg_list.append(item.bg_list)
        ed_list.append(item.ed_list)

    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
    input_masks = torch.tensor(input_masks, dtype=torch.long).cuda()
    subword_masks = torch.tensor(subword_masks, dtype=torch.long).cuda()
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).cuda()
    label_ids = torch.tensor(label_ids, dtype=torch.long).cuda()
    ner_ids = torch.tensor(ner_ids, dtype=torch.long).cuda()
    pos_ids = torch.tensor(pos_ids, dtype=torch.long).cuda()
    deprel_ids = torch.tensor(deprel_ids, dtype=torch.long).cuda()
    adjs = torch.tensor(adjs, dtype=torch.float).cuda()
    dists = torch.tensor(dists, dtype=torch.long).cuda()
    subj_poses = torch.tensor(subj_poses, dtype=torch.long).cuda()
    obj_poses = torch.tensor(obj_poses, dtype=torch.long).cuda()

    return input_ids, input_masks, subword_masks, segment_ids, label_ids, ner_ids, pos_ids, deprel_ids, \
        adjs, dists, subj_poses, obj_poses, bg_list, ed_list, subj_type_ids, obj_type_ids