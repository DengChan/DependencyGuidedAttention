"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import json

from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper, golVars

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.bert import BertConfig, BertTokenizer
from data.dataset import RelationDataset, collate_fn

golVars._init()

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='saved_models/05',help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
config = BertConfig.from_pretrained(opt["model_name_or_path"],
                                    num_labels=len(constant.LABEL_TO_ID),
                                    cache_dir=opt["cache_dir"])
trainer = GCNTrainer(opt, config)
trainer.load(model_file)

# load vocab
tokenizer = BertTokenizer.from_pretrained(opt["model_name_or_path"],
                                          do_lower_case=opt["do_lower_case"],
                                          cache_dir=opt["cache_dir"])

golVars.set_value("OPT", opt)
golVars.set_value("TKZ", tokenizer)
golVars.set_value("OUTPUT_EXAMPLES", True)

# load data
test_dataset = RelationDataset(opt, mode="test", tokenizer=tokenizer)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=int(args.batch_size), collate_fn=collate_fn, shuffle=False)


helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
test_golds = []
test_iterator = tqdm(test_dataloader, desc="Iteration")
for i, batch in enumerate(test_iterator):
    preds, labels, probs, loss = trainer.predict(batch)
    test_golds += labels
    predictions += preds

predictions = [id2label[p] for p in predictions]
test_golds = [id2label[p] for p in test_golds]
p, r, f1 = scorer.score(test_golds, predictions)

class_score = scorer.every_score(test_golds, predictions)

fjson = open(args.data_dir+"/test.json", 'r')
origin_data = json.load(fjson)
fjson.close()
with open("test/eval_output.txt", 'a', encoding='utf-8') as f:
    f.write("True Label\tPrediction\tSubject\tObject\tSentence\n")
    for i in range(len(predictions)):
        if test_golds[i] != predictions[i]:
            ss = origin_data[i]['subj_start']
            se = origin_data[i]['subj_end']
            os = origin_data[i]['obj_start']
            oe = origin_data[i]['obj_end']

            token = origin_data[i]['token']
            subj = " ".join(token[ss:ss + 1])
            obj = " ".join(token[os:os + 1])
            sent = " ".join(token)
            f.write("{}\t{}\t{}\t{}\t{}\n".format(test_golds[i], predictions[i], subj, obj, sent))


with open("test/scores.txt", 'a', encoding='utf-8') as f:
    f.write("Label\tP Score\tR Score\t F1 Score\n")
    for k, v in class_score.items():
        f.write("{}\t{}\t{}\t{}\n".format(k, v[0], v[1], v[2]))


print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset, p, r, f1))

print("Evaluation ended.")

