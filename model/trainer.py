"""
A trainer class.
"""
from apex import amp
import torch

from model.DGA import DGAModel
from model.BERT import BertRE
from model.bert import AdamW, get_linear_schedule_with_warmup
from utils import constant, torch_utils


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch, step):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = param_group['lr'] * self.opt['lr_decay']
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class GCNTrainer(Trainer):
    def __init__(self, opt, config):
        self.opt = opt
        self.model = BertRE(opt, config)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()

        optimizer_grouped_parameters = self.get_params(self.opt["fintune_bert"])
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=opt["lr"], eps=opt["adam_epsilon"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=opt["warmup_steps"],
                                                         num_training_steps=opt["t_total"])
        # Prepare optimizer and schedule (linear warmup and decay)
        if opt["fp16"]:
            self.model, optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt["fp16_opt_level"])

        self.model.zero_grad()

    def update(self, batch, step):
        # step forward
        self.model.train()
        # self.optimizer.zero_grad()

        rel_loss, pooling_output = self.model(batch)
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            rel_loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        try:
            loss_val = rel_loss.item()
        except:
            print(rel_loss)
            raise ValueError("ERROR")
        # backward
        if self.opt["fp16"]:
            with amp.scale_loss(rel_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            rel_loss.backward()

        if (step + 1) % self.opt["gradient_accumulation_steps"] == 0:
            if self.opt["fp16"]:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.opt['max_grad_norm'])
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])

        self.scheduler.step()
        self.optimizer.step()
        self.model.zero_grad()
        # print(self.optimizer.param_groups)
        return loss_val

    def predict(self, batch, unsort=True):
        # forward
        self.model.eval()
        loss, _ = self.model(batch)
        predictions, probs = self.model.predict()
        labels = self.model.get_labels()

        return predictions, labels, probs, loss.item()

    def get_params(self, fintune=True):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [(n, p) for n, p in self.model.Decoder.named_parameters()]
        if fintune:
            params += [(n, p) for n, p in self.model.Encoder.named_parameters()]

        optimizer_grouped_parameters = [
            {"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
             "weight_decay": self.opt["weight_decay"]},
            {"params": [p for n, p in params if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        return optimizer_grouped_parameters