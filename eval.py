"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import json

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='saved_models/00',help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

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
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions)

class_score = scorer.every_score(batch.gold(), predictions)

fjson = open(data_file, 'r')
origin_data = json.load(fjson)
fjson.close()
with open("eval_output.txt", 'a') as f:
    f.write("True Label\tPrediction\tSubject\tObject\tSentence\n")
    for i in range(len(predictions)):
        if batch.gold()[i] != predictions[i]:
            ss = origin_data[i]['subj_start']
            se = origin_data[i]['subj_end']
            os = origin_data[i]['obj_start']
            oe = origin_data[i]['obj_end']

            token = origin_data[i]['token']
            subj = " ".join(token[ss:ss + 1])
            obj = " ".join(token[os:os + 1])
            sent = " ".join(token)
            f.write("{}\t{}\t{}\t{}\t{}\n".format(batch.gold()[i], predictions[i], subj, obj, sent))


with open("scores.txt", 'a', encoding='utf-8') as f:
    f.write("Label\tP Score\tR Score\t F1 Score\n")
    for k, v in class_score.items():
        f.write("{}\t{}\t{}\t{}\n".format(k, v[0], v[1], v[2]))


print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset, p, r, f1))

print("Evaluation ended.")

