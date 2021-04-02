import os
import torch
import pandas as pd
import numpy as np
import dill
from functools import partial, reduce
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
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
from collections import Counter
from gensim.models import Word2Vec
import multiprocessing

class Vocab(object):
    def __init__(self, special=['__eou__', '__eot__'],
                 lower_case=True,
                 model_name=None,
                 bert_path='',
                 worddict={}):

        self.worddict = worddict
        if bert_path:
            self.load_vocab(os.path.join(bert_path, 'vocab.txt'))
            self_n_bert_token = len(self.worddict)
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

    def read_csv_file(self, path, train, func=None):
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
            # self.build_worddict(*data)
        else:
            neg_samples = header[2:]
            neg_data = [f[neg].tolist() for neg in neg_samples]
            data = (context, response, neg_data)
        return self.iter_data(data, task, train, func)

    def load_vocab(self,
                   vocab_file):
        with open(vocab_file, "r", encoding='utf8') as f:
            for idx, word in enumerate(f):
                self.worddict[word[:-1]] = idx

    def get_word(self, c, r, l):
        words = []
        [words.extend(sentence) for sentence in c]
        [words.extend(sentence) for sentence in r]
        return words
    # def build_worddict(self, c, r, l):

    #     words = self.get_word(c, r, l) # TODO need to add words from train set when test

    #     counts = Counter(words)
    #     num_words = len(counts)
    #     self.offset = 0

    #     # self.worddict["__pad__"] = 0
    #     # self.worddict["__oov__"] = 1
    #     #
    #     # for t in self.special:
    #     #     self.worddict[t] = offset
    #     #     offset += 1
    #     n_tokens = len(self.worddict)
    #     for i, word in enumerate(counts.most_common(num_words)):
    #         if word[0] not in self.worddict:
    #             self.worddict[word[0]] = n_tokens
    #             n_tokens += 1
    #             # self.tokenizer.add_tokens([word[0]])

    # def words_to_indices(self, text):
    #     """
    #     transform sentence to indices
    #     :param text: sentence
    #     :return: list of indices
    #     """
    #     indices = []
    #     for word in text:
    #         if word in self.worddict:
    #             index = self.worddict[word]
    #         else:
    #             index = self.worddict["[UNK]"]
    #         indices.append(index)
    #     return indices

    # def file_to_indices(self, data, train):
    #     task = self.progress.add_task("test", dataset="word2indices", start=False)
    #     return self.iter_data(data, task, train, self.words_to_indices)

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

    def train_word2vec(self, data):
        data_indices_str = []
        for d in data[:2]:
            for x in d:
                data_indices_str.append(list(map(str, x)))
        w2v_model = Word2Vec(data_indices_str, vector_size=300, window=5, min_count=5,
                             sg=1, workers=multiprocessing.cpu_count())
        return w2v_model

    def build_embed_layer(self, embeddings_file):
        embed_dict = {}
        with open(embeddings_file, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx == 0: continue
                line = line.split()
                try:
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embed_dict[word] = line[1:]
                except ValueError:
                    continue
            n_token = len(self.worddict)
            embed_dim = len(list(embed_dict.values())[0])
            embed_matrix = np.zeros((n_token, embed_dim))
            for word, i in self.worddict.items():
                if str(i) in embed_dict:
                    embed_matrix[i] = np.array(embed_dict[str(i)], dtype=float)
                else:
                    if word == "[PAD]":
                        continue

                    embed_matrix[i] = np.random.normal(size=(embed_dim))
        return embed_matrix
    # https://github.com/coetaur0/ESIM/blob/65611601ff9f17f76e1f246e8e46b5fc4bee13fc/esim/data.py
    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.
        Args:
            embeddings_file: A file containing pretrained word embeddings.
        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix



class UbuntuCorpus(Dataset):

    def __init__(self, path, type, save_path, **kwargs):
        self.path = path
        self.type = type
        self.dataset = None
        self.Vocab = Vocab(**kwargs)

        assert os.path.exists(save_path)
        s_path = os.path.join(save_path, type)
        self.data = self.load_data(s_path)

        if not self.data:
            self.data = self.Vocab.read_csv_file(path=path, train=(type=='train'))
            # word2vec
            # wordict_path = os.path.join(save_path, 'worddict')
            # if os.path.exists(wordict_path):
            #     self.Vocab.worddict = self.load(wordict_path)
            # elif type == 'train' and self.Vocab.worddict:
            #         self.dump(self.Vocab.worddict, os.path.join(save_path, 'worddict'))
            # # build_worddict first
            # assert self.Vocab.worddict != {}
            self.dump(self.data, s_path)
        if type == 'train':
            # if not self.Vocab.worddict: self.Vocab.worddict = self.load(os.path.join(save_path, 'worddict'))
            emb_path = reduce(os.path.join, [save_path, 'embeddings', 'ubuntu_corpus.txt'])
            bert_emb_path = reduce(os.path.join, [save_path, 'embeddings', 'ubuntu_corpus.npy'])
            if not os.path.exists(emb_path):
                self.embeddings = self.Vocab.train_word2vec(self.data)
                self.embeddings.wv.save_word2vec_format(emb_path, binary=False)
            if not os.path.exists(bert_emb_path):
                self.embeddings = self.Vocab.build_embed_layer(emb_path)
                np.save(bert_emb_path, self.embeddings)
            else:
                self.embeddings = np.load(bert_emb_path)
                # self.embeddings = Word2Vec.load(emb_path) # offset = 4



    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        # input : context, response, label/ neg_samples
        # output: (context, context_len), (response, response_len), ...
        if self.type == 'train':
            return (torch.tensor(self.data[0][idx], dtype=torch.long), len(self.data[0][idx])), (torch.tensor(self.data[1][idx], dtype=torch.long), len(self.data[1][idx])), self.data[2][idx]
        else:
            n_sample = len(self.data[2])
            neg_line = []
            for neg in range(n_sample):
                neg_line.append((torch.tensor(self.data[2][neg][idx]), len(self.data[2][neg][idx])))
            return (torch.tensor(self.data[0][idx]), len(self.data[0][idx])), (torch.tensor(self.data[1][idx]), len(self.data[1][idx])), neg_line

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
    t_c, t_r, label = zip(*data)
    padded_c, padded_r = padding(*zip(*t_c)), padding(*zip(*t_r))
    return padded_c, padded_r, label

def ub_corpus_test_collate_fn(data):

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
    save_path = '/remote_workspace/fusion_esim/data/examples'
    model_name = 'bert'
    bert_path = '/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-8_H-512_A-8'
    # val_dataset = UbuntuCorpus(path=path, type='valid', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'])
    train_dataset = UbuntuCorpus(path=train_path, type='train', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
    # val_dataloader = DataLoader(val_dataset, batch_size=3, collate_fn=ub_corpus_test_collate_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=ub_corpus_train_collate_fn)
    # while 1:
    #
    #     for i, (c, s, n) in enumerate(val_dataloader):
        # for i, d in enumerate(train_dataloader):
        #     print(d)
        #     exit()
    
        # print("end epoch")
