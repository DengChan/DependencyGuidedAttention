"""
一些注意事项
1. 矩阵必须在这生成，因为head subj_pos 等 可能会被截断，
所以先生成完整矩阵，如果需要截断，直接取矩阵的子矩阵即可，
防止生产邻接矩阵时找不到头节点


TO DO:
1. 两种处理subword的方式
    (1) subword 当作子节点
    (2) subword 等同原节点，子节点将会有多个父节点,
        还要一版计算LCA dist的代码, 实现思路：用原来的老head生成一波dist，subword的dist等于这个dist中原来单词的位置的值
"""

import os
import numpy as np
import json
import torch
from utils import constant
from model.tree import head_to_tree, tree_to_adj
from torch.utils.data import TensorDataset

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
        if opt['lower']:
            self.words = [t.lower() for t in self.words]
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ner_ids, pos_ids, deprel_ids, heads, adj, dists):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ner_ids = ner_ids
        self.pos_ids = pos_ids
        self.deprel_ids = deprel_ids
        self.heads = heads
        self.adj = adj
        self.dists = dists


def read_examples_from_file(opt, data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        all_data = json.load(f)
        for single_data in all_data:
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
        for word, pos, ner, deprel in zip(example.words, example.pos, example.ner, example.deprel):
            # 更新old2new的映射
            old_index2new_index[idx] = len(tokens)

            word_tokens = tokenizer.tokenize(word)
            word_tokens_tmp.append(word_tokens)
            tokens.extend(word_tokens)
            # 更新old_end2new_end的映射
            old_end2new_end[idx] = len(tokens) - 1

            # 被拆分的保存同样的pos和ner
            pos_ids.extend([constant.POS_TO_ID[pos]] * len(word_tokens))
            ner_ids.extend([constant.NER_TO_ID[ner]] * len(word_tokens))
            # 如果单词被拆分，第一部分的头部为原来， 拆开的剩下的单词，以第一个单词为头
            deprel_ids.extend([constant.DEPREL_TO_ID[deprel]]+[constant.DEPREL_TO_ID[constant.SAME_TOKEN]] * (len(word_tokens) - 1))
            idx += 1

        assert len(example.head) == len(word_tokens_tmp)
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
        else:


        # 检查tokenzier后是否还能对齐
        assert len(tokens) == len(example.words)

        # 处理subj_position 和 obj_position
        l = len(tokens)
        subj_start = old_index2new_index[example.subj_start]
        subj_end = old_end2new_end[example.subj_end]
        obj_start = old_index2new_index[example.obj_start]
        obj_end = old_end2new_end[example.obj_end]
        subj_pos = get_positions(subj_start, subj_end, l)
        obj_pos = get_positions(obj_start, obj_end, 1)

        special_tokens_count = 3 if sep_token_extra else 2
        # 生成邻接矩阵,
        # max(l, max_seq_length-2) ： 如果 l> max_seq_length-2保证矩阵是完整的， 如果l < max_seq_length-2, 保证矩阵长度max_seq_length-2
        # len(dists) == l 仍需要pad
        adj, dists = inputs_to_tree_reps(heads, tokens, l, opt['prune_k'], subj_pos, obj_pos,
                                         max(l, max_seq_length-special_tokens_count))

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
            heads = heads[:(max_seq_length - special_tokens_count)]
            subj_pos = subj_pos[:(max_seq_length - special_tokens_count)]
            obj_pos = obj_pos[:(max_seq_length - special_tokens_count)]
            dists = dists[:(max_seq_length - special_tokens_count)]

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
        else:
            tokens = [cls_token] + tokens
            ner_ids = [constant.PAD_ID] + ner_ids
            pos_ids = [constant.PAD_ID] + pos_ids
            deprel_ids = [constant.PAD_ID] + deprel_ids
            heads = [-1] + heads
            subj_pos = [subj_pos[0]-1] + subj_pos
            obj_pos = [obj_pos[0]-1] + obj_pos
            dists = [0] + dists
            segment_ids = [cls_token_segment_id] + segment_ids

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
            heads = ([-1] * padding_length) + heads
            dists = ([0] * padding_length) + dists
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
            heads += ([-1] * padding_length)
            dists += ([0] * padding_length)
            for i in range(padding_length):
                subj_pos += [subj_pos[-1] + 1]
                obj_pos += [obj_pos[-1] + 1]

        # 单独处理 adj
        adj = np.pad(adj, ((1, 1), (1, 1)), 'constant', constant_values=(0.0, 0.0))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ner_ids) == max_seq_length
        assert len(subj_pos) == max_seq_length
        assert len(dists) == max_seq_length
        assert len(adj.shape[0]) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("ner_ids: %s", " ".join([str(x) for x in ner_ids]))
            logger.info("subj_position: %s", " ".join([str(x) for x in subj_pos]))
            logger.info("LCA distances: %s", " ".join([str(x) for x in dists]))
            logger.info("Adj(front 10 node): {}".format(adj[:10, :10]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              ner_ids=ner_ids,
                              pos_ids=pos_ids,
                              deprel_ids=deprel_ids,
                              heads=heads,
                              adj=adj,
                              dists=dists))
    return features


def load_and_cache_examples(opt, tokenizer, labels, pad_token_label_id, mode):
    cached_features_file = os.path.join(opt["data_dir"], "cached_{}.data".format(mode))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", opt["data_dir"])
        examples = read_examples_from_file(opt, opt["data_dir"], mode)
        features = convert_examples_to_features(examples, labels, opt["max_seq_length"], tokenizer,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_label_id=pad_token_label_id
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ner_ids = torch.tensor([f.ner_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_deprels_ids = torch.tensor([f.deprel_ids for f in features], dtype=torch.long)
    all_heads = torch.tensor([f.heads for f in features], dtype=torch.long)
    all_adj = torch.tensor([f.adj for f in features], dtype=torch.float)
    all_dists = torch.tensor([f.dists for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id,
                            all_ner_ids, all_pos_ids, all_deprels_ids, all_heads, all_adj, all_dists)
    return dataset


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, maxlen):
    tree, dist = head_to_tree(head, words, l, prune, subj_pos, obj_pos)
    # adj 邻接边为边类型
    adj = tree_to_adj(maxlen, tree)

    return adj, dist


def heads_to_adj(heads, maxlen, only_child=False, self_loop=True):
    """ 每一个节点可能有多个head"""
    ret = np.zeros((maxlen, maxlen), dtype=np.float32)
    for i in range(len(heads)):
        hs = heads[i]
        for h in hs:
            if h == 0:
                continue
            ret[h-1][i] = 1

    if not only_child:
        ret = ret + ret.T
    if self_loop:
        for i in range(len(heads)):
            ret[i, i] = 1
    return ret