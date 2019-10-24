# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MyLoss(nn.Module):
    def __init__(self, label_num, size_average=True):
        super(MyLoss, self).__init__()
        self.label_num = label_num

    def forward(self, scores, labels):
        labels_mask = torch.Tensor(np.eye(self.label_num)[labels]).byte() # [B X L] // One hot 形式
        negtive_mask = torch.Tensor(labels).eq(0).byte() # [B]    // 负样例的mask为1
        print(labels_mask)
        print(negtive_mask)
        positive_scores = scores.masked_select(labels_mask).masked_select(negtive_mask.eq(0)) # 每一行选出label位置的score, 在把非负样例的得分挑出来
        positive_loss = (1-torch.nn.functional.sigmoid(positive_scores)).log().sum()

        # 除标签外 最大的得分
        wrong_scores = scores.masked_fill(labels_mask, 0) # [B x L]
        wrong_scores, _ = wrong_scores.max(-1) # [B]
        negtive_loss = torch.nn.functional.sigmoid(wrong_scores).log().sum()

        loss = positive_loss + negtive_loss

        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.from_numpy(alpha)
                self.alpha = Variable(self.alpha)
        self.gamma = gamma*1.0
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)].float()
        # print("========================\nalpha: =====================")
        # print(alpha)
        # print("======================================================")
        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1-probs), self.gamma))
        batch_loss = batch_loss*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss