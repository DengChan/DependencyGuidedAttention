# -*- coding: utf-8 -*-

pred = []
label = []
with open("answer_key1.txt") as f:
    for line in f.readlines():
        pred.append(line.strip().split('\t')[1])

cnt = 0
with open("proposed_answer1.txt") as f:
    for line in f.readlines():
        cnt += 1
        try:
            label.append(line.strip().split('\t')[1])
        except:
            print(cnt)
from utils.scorer import semeval_score

semeval_score(label, pred)
