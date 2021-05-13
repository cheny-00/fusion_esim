import os
import torch
import pandas as pd
import numpy as np
import dill
from functools import partial, reduce
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertTokenizerFast,
    DistilBertModel,
)
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
import spacy
from gensim.models import Word2Vec
import multiprocessing

class Vocab(object):
    def __init__(self, special=['__eou__', '__eot__'],
                 lower_case=True,
                 model_name=None,
                 bert_path='',
                 worddict={}):

        self.worddict = worddict
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[dataset]}", justify="right"),
            BarColumn(bar_width=100, style="black on white"),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(),
            "/",
            TimeElapsedColumn(),
        )
        self.special = special
        self.offset = 0

        self.lower_case = lower_case
        self.model_name = model_name
        if 'distilbert' in model_name:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(bert_path if bert_path else model_name,
                                                                     do_lower_case=lower_case)
        elif 'bert' in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path if bert_path else model_name,
                                                           do_lower_case=lower_case)
            self.n_bert_token = self.tokenizer.vocab_size


    def pre_tokenize(self, line):
        for token in self.special:
            line = line.replace(token, "")
        return line
    def tokenize(self, line): #TODO remove special tokens, split turn

        encoded_line = self.tokenizer.encode(self.pre_tokenize(line), add_special_tokens=True, truncation=True)
        # TODO add_special_tokens & truncation
        return encoded_line


    def read_csv_file(self, path, train, func=None, w2v_f = None):
        assert os.path.exists(path)
        if not func:
            func = self.tokenize
        type = 'train' if train else 'valid or test'
        task = self.progress.add_task(type, dataset="UbuntuCorpus/{}".format(type), start=False)
        f = pd.read_csv(path)
        header = f.columns
        context = f[header[0]].tolist()
        response = f[header[1]].tolist()
        if train:
            label = f[header[2]]
            data = (context, response, label)
        else:
            neg_samples = header[2:]
            neg_data = [f[neg].tolist() for neg in neg_samples]
            data = (context, response, neg_data)
        b_data = self.iter_data(data, task, train, func)
        return b_data


    def get_word(self, c, r, l):
        words = []
        [words.extend(sentence) for sentence in c]
        [words.extend(sentence) for sentence in r]
        return words

    def build_worddict(self, c, r, l):

        words = self.get_word(c, r, l) # TODO need to add words from train set when test

        counts = Counter(words)
        num_words = len(counts)
        self.offset = 0
        for i, word in enumerate(counts.most_common(num_words)):
            if word[0] not in self.worddict and 5 <= word[1]:
                self.worddict[word[0]] = word[1]


    def iter_data(self, data, task, train, func):
        c, r = data[0], data[1]
        c_res, r_res = [], []
        if train:
            l = data[2]
        else:
            n_samples = data[2]
            n_res = [[] for _ in range(len(n_samples))]
        with self.progress:
            self.progress.update(task, total=len(c))
            self.progress.start_task(task)
            for idx in range(len(c)):
                c_res.append(func(c[idx]))
                r_res.append(func(r[idx]))
                if not train:
                    for i, neg in enumerate(n_samples):
                        n_res[i].append(func(neg[idx]))
                self.progress.update(task, advance=1)
        return c_res, r_res, l if train else n_res

    def build_embed_layer(self, embedding_weight):
        res_embed = torch.randn_like(embedding_weight)
        for word, times in self.worddict.items():
            res_embed[word] = embedding_weight[word]
        return res_embed




class UbuntuCorpus(Dataset):

    def __init__(self, path, type, save_path, **kwargs):

        self.path = path
        self.type = type
        self.dataset = None
        self.Vocab = Vocab(**kwargs)

        assert os.path.exists(save_path)
        s_path = os.path.join(save_path, type)
        self.data = self.load_data(s_path)
        wordict_path = os.path.join(save_path, 'worddict')

        if not self.data:
            self.data = self.Vocab.read_csv_file(path=path, train=(type=='train')) # (w2v, bert)
            # word2vec
            if os.path.exists(wordict_path):
                self.Vocab.worddict = self.load(wordict_path)
            elif type == 'train':
                self.Vocab.build_worddict(*self.data)
                self.dump(self.Vocab.worddict, os.path.join(save_path, 'worddict'))
            # build_worddict first
            assert self.Vocab.worddict, 'make suere worddict exist'
            self.dump(self.data, s_path)



    def __len__(self):
        return len(self.data[0])
    @staticmethod
    def get_item(idx, data, train):
        if train:
            return (torch.tensor(data[0][idx], dtype=torch.long), len(data[0][idx])),\
                   (torch.tensor(data[1][idx], dtype=torch.long), len(data[1][idx])),\
                   data[2][idx]
        else:
            n_sample = len(data[2])
            neg_line = []
            for neg in range(n_sample):
                neg_line.append((torch.tensor(data[2][neg][idx]), len(data[2][neg][idx])))
            return (torch.tensor(data[0][idx]), len(data[0][idx])),\
                   (torch.tensor(data[1][idx]), len(data[1][idx])),\
                   neg_line
    def __getitem__(self, idx):
        # input : context, response, label/ neg_samples
        # output: (context, context_len), (response, response_len), ...
        return self.get_item(idx, self.data, self.type=='train')

    def dump(self, data, path):
        with open(path, 'wb') as f:
            dill.dump(data, f)

    def load_data(self, path):
        data = []
        if os.path.exists(path):
            data = self.load(path)
        return data

    def load(self, path):
        with open(path, 'rb') as f:
            data = dill.load(f)
        return data


def ub_corpus_train_collate_fn(data):
    if len(data[0]) == 2:
        data = tuple(zip(*data))
        return (ub_corpus_train_collate_fn(data[0]), ub_corpus_train_collate_fn(data[1]))
    t_c, t_r, label = zip(*data)
    padded_c, padded_r = padding(*zip(*t_c)), padding(*zip(*t_r))
    return padded_c, padded_r, label

def ub_corpus_test_collate_fn(data):

    if len(data[0]) == 2:
        data = tuple(zip(*data))
        return (ub_corpus_test_collate_fn(data[0]), ub_corpus_test_collate_fn(data[1]))
    # data : [((c, c_l), (r, r_l), ((neg_1, neg_1_l), ...(neg_n, neg_n_l))) * batch_size]
    t_c, t_r, n_s = zip(*data)
    neg = list(zip(*n_s)) # transpose n_s
    neg_col = [tuple(zip(*n)) for n in neg] # (neg_1 * batch_size), (neg_1_len * batch_size)
    padded_c, padded_r = padding(*zip(*t_c)), padding(*zip(*t_r))
    # padded_neg = list(map(padding, neg_col)) # 蜜汁bug ？？？
    padded_neg = neg_part(neg_col)
    return padded_c, padded_r, padded_neg

def neg_part(negs):
    res = []
    for seq, len in negs:
        res.append(padding(seq, len))
    return res
def padding(seqs, seq_lens):
    padded_seq = torch.zeros(len(seqs), max(seq_lens)).long()
    for i, seq in enumerate(seqs):
        end = seq_lens[i]
        padded_seq[i, :end] = seq[:end]
    return (padded_seq, torch.tensor(seq_lens))





if __name__ == '__main__':

    path = '/remote_workspace/dataset/default/valid.csv'
    train_path = '/remote_workspace/dataset/default/train.csv'
    save_path = '/remote_workspace/fusion_esim/data/bert'
    model_name = 'bert'
    bert_path = '/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-8_H-512_A-8'
    # val_dataset = UbuntuCorpus(path=path, type='valid', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
    train_dataset = UbuntuCorpus(path=train_path, type='train', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
    # val_dataloader = DataLoader(val_dataset, batch_size=5, collate_fn=ub_corpus_test_collate_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=ub_corpus_train_collate_fn)
    print()
    # while 1:
    # #
    #     # for i, (c, s, n) in enumerate(val_dataloader):
    #     for i, d in enumerate(val_dataloader):
    #         print(len(d))
    #         print(len(d[0]))
    #         print(len(d[0][2]))
    #         w, b = d[0][2], d[1][2]
    #         for wn, bn in zip(w, b):
    #             print(len(wn[0]), len(wn[1]))
    #             print(len(bn[0]), len(bn[1]))
    #         # print(d)
    #         exit()
    
        # print("end epoch")
