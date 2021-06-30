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

        self.do_eou = True
        self.do_eot = True

        self.max_context_len = 280
        self.max_response_len = 40

        self.lower_case = lower_case
        self.model_name = model_name
        if 'distilbert' in model_name:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(bert_path if bert_path else model_name,
                                                                     do_lower_case=lower_case)
        elif 'bert' in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path if bert_path else model_name,
                                                           do_lower_case=lower_case)
        self.n_bert_token = self.tokenizer.vocab_size
        if self.do_eot:
            self.tokenizer.add_tokens(["[EOU]", "[EOT]"])

    def pre_tokenize(self, line, iscontext):
        replace_token = {"__eou__":"[EOU]",
                         "__eot__": "[EOT]"}
        sp = self.special
        line = line.replace("__eou__", "[EOU]")
        line = line.replace("__eot__", "[EOT]")
        # line = [replace_token[x] if x in sp else x for x in line]

        max_len, rm_pos = (self.max_context_len, 0) if iscontext else (self.max_response_len, -1)
        line = self.tokenizer.tokenize(line)
        while len(line) > max_len - rm_pos - 2:
            line.pop(rm_pos)
        # for token in self.special:
        #     line = line.replace(token, "")
        return line
    def tokenize(self, line, iscontext): #TODO remove special tokens, split turn
        encoded_line = self.tokenizer.convert_tokens_to_ids(self.pre_tokenize(line, iscontext))
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
                c_res.append(func(c[idx], True))
                r_res.append(func(r[idx], False))
                if not train:
                    for i, neg in enumerate(n_samples):
                        n_res[i].append(func(neg[idx], False))
                self.progress.update(task, advance=1)
        return c_res, r_res, l if train else n_res

    def build_embed_layer(self, embedding_weight):
        res_embed = torch.randn_like(embedding_weight)
        for word, times in self.worddict.items():
            res_embed[word] = embedding_weight[word]
        return res_embed



import pickle

class UbuntuCorpus(Dataset):

    def __init__(self, path, type, save_path, **kwargs):

        self.path = path
        self.type = type
        self.dataset = None
        self.max_context_len = 280
        self.max_response_len = 40
        self.Vocab = Vocab(**kwargs)
        hm = ""


        assert os.path.exists(save_path)
        s_path = os.path.join(save_path, type)
        self.data = self.load_data(s_path)
        if type == 'train':
            self.e_data = self.pickle_load_data(hm + '/remote_workspace/fusion_esim/data/w2v/esim_data.pkl')
        else:
            self.e_data = self.pickle_load_data(hm + '/remote_workspace/fusion_esim/data/w2v/esim_{}_data.pkl'.format(type))
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
            assert self.Vocab.worddict, 'make sure worddict exist'
            self.dump(self.data, s_path)

    @staticmethod
    def pickle_load_data(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.data[0])

    @staticmethod
    def get_item(idx, data, e_data, train, max_context_len, max_response_len):
        context, response = data[0][idx], data[1][idx]
        anno_seq, seg_ids, attn_mask = bert_input_data(context, response, max_context_len, max_response_len)
        if train:
            label = data[2][idx]
            features = dict()
            features["esim_data"] = ((torch.tensor(e_data['context'][idx], dtype=torch.long), e_data['c_len'][idx]),
                                     (torch.tensor(e_data['response'][idx], dtype=torch.long), e_data['r_len'][idx]))
            features["anno_seq"] = torch.tensor(anno_seq, dtype=torch.long)
            features["seg_ids"] = torch.tensor(seg_ids, dtype=torch.long)
            features["attn_mask"] = torch.tensor(attn_mask, dtype=torch.long)
            features["label"] = torch.tensor(label, dtype=torch.long)
            return features

        n_sample = len(data[2])
        neg_line = list()
        features = dict()
        features["anno_seq"] = [torch.tensor(anno_seq, dtype=torch.long)]
        features["seg_ids"] = [torch.tensor(seg_ids, dtype=torch.long)]
        features["attn_mask"] = [torch.tensor(attn_mask, dtype=torch.long)]

        for neg in range(n_sample):
            neg_seq = data[2][neg][idx]
            anno, seg, attn = bert_input_data(context, neg_seq, max_context_len, max_response_len)
            features["anno_seq"].append(torch.tensor(anno, dtype=torch.long))
            features["seg_ids"].append(torch.tensor(seg, dtype=torch.long))
            features["attn_mask"].append(torch.tensor(attn, dtype=torch.long))
            neg_line.append((torch.tensor(anno[max_context_len:], dtype=torch.long), len(neg_seq)))
        features["esim_data"] = ((torch.tensor(e_data['context'][idx], dtype=torch.long), e_data['c_len'][idx]),
                                 (torch.tensor(e_data['response'][idx], dtype=torch.long),e_data['r_len'][idx]),
                                 (torch.tensor(e_data['neg'][idx], dtype=torch.long), e_data['neg_len'][idx]))
        return features

    @staticmethod
    def get_bert_item(idx, data, train, max_context_len, max_response_len):
        context, response = data[0][idx], data[1][idx]
        anno_seq, seg_ids, attn_mask = bert_input_data(context, response, max_context_len, max_response_len)
        if train:
            label = data[2][idx]
            features = dict()
            features["esim_data"] = ((torch.tensor(anno_seq[:max_context_len], dtype=torch.long), len(context)),
                                     (torch.tensor(anno_seq[max_context_len:], dtype=torch.long), len(response)),
                                     label)
            features["anno_seq"] = torch.tensor(anno_seq, dtype=torch.long)
            features["seg_ids"] = torch.tensor(seg_ids, dtype=torch.long)
            features["attn_mask"] = torch.tensor(attn_mask, dtype=torch.long)
            features["label"] = label
            return features

        n_sample = len(data[2])
        neg_line = list()
        features = dict()
        features["anno_seq"] = [torch.tensor(anno_seq, dtype=torch.long)]
        features["seg_ids"] = [torch.tensor(seg_ids, dtype=torch.long)]
        features["attn_mask"] = [torch.tensor(attn_mask, dtype=torch.long)]

        for neg in range(n_sample):
            neg_seq = data[2][neg][idx]
            anno, seg, attn = bert_input_data(context, neg_seq, max_context_len, max_response_len)
            features["anno_seq"].append(torch.tensor(anno, dtype=torch.long))
            features["seg_ids"].append(torch.tensor(seg, dtype=torch.long))
            features["attn_mask"].append(torch.tensor(attn, dtype=torch.long))
            neg_line.append((torch.tensor(anno[max_context_len:], dtype=torch.long), len(neg_seq)))
        features["esim_data"] = ((torch.tensor(anno_seq[:max_context_len], dtype=torch.long), len(context)),
                                 (torch.tensor(anno_seq[max_context_len:], dtype=torch.long), len(response)),
                                 neg_line)
        return features
    def __getitem__(self, idx):
        # input : context, response, label/ neg_samples
        # output: (context, context_len), (response, response_len), ...
        return self.get_bert_item(idx, self.data, self.type=='train', self.max_context_len, self.max_response_len)
        # else: return self.get_item(idx, self.data, self.e_data, self.type=='train', self.max_context_len, self.max_response_len)

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

def max_trim(text, max_len, rm_pos):
    while max_len < len(text):
        text.pop(rm_pos)
    return text

def bert_input_data(context, response, max_context_len=280, max_response_len=40, PAD=0, CLS=101, SEP=102):

    context, response = max_trim(context, max_context_len - 2, 0), max_trim(response, max_response_len - 1, -1)
    context = [101] + context + [102]
    segment_ids = [0] * max_context_len
    attn_mask = [1] * len(context)

    while len(context) < max_context_len:
        context.append(0)
        attn_mask.append(0)
    assert len(context) == len(segment_ids) == len(attn_mask)

    response = response + [102]
    segment_ids.extend([1] * len(response))
    attn_mask.extend([1] * len(response))

    while len(response) < max_response_len:
        response.append(0)
        segment_ids.append(0)
        attn_mask.append(0)
    context_response = context + response

    assert len(context_response) == len(segment_ids) == len(attn_mask)
    return context_response, segment_ids, attn_mask


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
    save_path = '/remote_workspace/fusion_esim/data/bert_with_eot'
    model_name = 'bert'
    bert_path = '/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-12_H-768_A-12'
    train_dataset = UbuntuCorpus(path=train_path, type='train', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
    val_dataset = UbuntuCorpus(path=path, type='valid', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=8)
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8)
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
