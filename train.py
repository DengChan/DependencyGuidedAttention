"""
Train a model on TACRED.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm

from model.trainer import GCNTrainer
from model.bert import BertConfig, BertTokenizer
from utils import torch_utils, scorer, constant, helper
from data.dataset import RelationDataset, collate_fn

from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss

import logging


parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", default="",
                    type=str, help="Path to pre-trained model")
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")

parser.add_argument('--data_dir', type=str, default='dataset/')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
# Input
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=50, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=50, help='POS embedding dimension.')
parser.add_argument('--dist_dim', type=int, default=56, help='LCA distance embedding dimension.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='Input dropout rate.')


# Attention
parser.add_argument('--K_dim', type=int, default=64, help='K dimension.')
parser.add_argument('--V_dim', type=int, default=64, help='V dimension.')
parser.add_argument('--num_heads', type=int, default=4, help='num of heads')
parser.add_argument('--feedforward_dim', type=int, default=512, help='feedforward dim')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden state size.')
parser.add_argument('--num_layers', type=int, default=6, help='Num of Sequence Encoder layers.')
parser.add_argument('--dep_layers', type=int, default=0, help='Num of Dependency Encoder layers.')


parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=7000, help='Only finetune top N word embeddings.')
parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")


parser.add_argument('--entity_mask', type=bool, default=False,
                    help="use ner kind to mask entity word")
parser.add_argument('--self_loop', type=bool, default=True,
                    help="use ner kind to mask entity word")
parser.add_argument('--first_subword_ner', type=bool, default=False,
                    help="only tag first subword ner, left subword is masked")
parser.add_argument('--subword_to_children', type=bool, default=True,
                    help="treat subword to first subword's child in dep tress")
parser.add_argument('--only_child', type=bool, default=False, help="whether use double direction edge.")
parser.add_argument('--deprel_edge', type=bool, default=False, help="whether use deprel info on edge")
parser.add_argument('--prune_k', default=-1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')


parser.add_argument('--conv_l2', type=float, default=0.0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0.0, help='L2-penalty for all pooling output.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument('--lr', type=float, default=0.00002, help='Applies to sgd and adagrad.')
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")


parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Gradient clipping.')

parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=30, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='18', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument("--fp16", type=bool, default=False,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)
opt["model_name_or_path"] = args.model_name_or_path
# load vocab

# ner pad token id for unit extraction
pad_token_label_id = CrossEntropyLoss().ignore_index

# load config and tokenzier
config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                    num_labels=len(constant.LABEL_TO_ID),
                                    cache_dir=args.cache_dir if args.cache_dir else None)
tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                          do_lower_case=args.do_lower_case,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))

train_dataset = RelationDataset(opt, mode="train", tokenizer=tokenizer, pad_token_ner_label_id=pad_token_label_id)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              collate_fn=collate_fn)
t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epoch
opt["t_total"] = t_total

dev_dataset = RelationDataset(opt, mode="dev", tokenizer=tokenizer, pad_token_ner_label_id=pad_token_label_id)
dev_sampler = SequentialSampler(dev_dataset)
dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size,
                            collate_fn=collate_fn)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = GCNTrainer(opt, config)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt, config)
    trainer.load(model_file)

id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']
max_dev_scores = -1.0

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

writer = SummaryWriter()
# start training
train_iterator = trange(int(args.num_epoch), desc="Epoch")
for epoch in train_iterator:
    train_loss = 0
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    all_train_count = 0
    for i, batch in enumerate(epoch_iterator):
        start_time = time.time()
        loss = trainer.update(batch, global_step)
        bs = len(batch[-3])
        all_train_count += bs
        epoch_iterator.set_postfix(loss="{:.4f}".format(loss), mode="Train")
        train_loss += loss
        # train writer
        writer.add_scalar('loss/train', loss, global_step)
        global_step += 1

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    dev_iterator = tqdm(dev_dataloader, desc="Iteration")
    all_dev_count = 0
    dev_golds = []
    for i, batch in enumerate(dev_iterator):
        bs = len(batch[-3])
        all_dev_count += bs
        preds, labels, probs, loss = trainer.predict(batch)
        dev_golds += labels
        predictions += preds
        dev_loss += loss

    predictions = [id2label[p] for p in predictions]
    dev_golds = [id2label[p] for p in dev_golds]
    train_loss = train_loss / all_train_count # avg loss per batch
    dev_loss = dev_loss / all_dev_count

    dev_p, dev_r, dev_f1 = scorer.score(dev_golds, predictions)

    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,
                                                                                     train_loss, dev_loss, dev_f1))
    dev_score = dev_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss,
                                                                dev_score, max([dev_score] + dev_score_history)))

    # dev writer
    writer.add_scalar('loss/dev', dev_loss, epoch)
    # metrics writer
    writer.add_scalar('metrics/P', dev_p, epoch)
    writer.add_scalar('metrics/R', dev_r, epoch)
    writer.add_scalar('metrics/F1', dev_f1, epoch)

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    if epoch == 1 or dev_score > max_dev_scores:
        max_dev_scores = dev_score
        trainer.save(model_save_dir + '/best_model.pt', epoch)
        config.save_pretrained(model_save_dir)
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    if epoch % opt['save_epoch'] == 0:
        trainer.save(model_file, epoch)
        config.save_pretrained(model_save_dir)

print("Training ended.")

