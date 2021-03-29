
import os
import dill
import time
from torchtext import data
from torchtext.vocab import GloVe
from spacy.attrs import ORTH, NORM
import spacy



def get_train_fields(TRAIN_TEXT, LABEL):
    train_fields = [
        ('Context', TRAIN_TEXT),
        ('Utterance', TRAIN_TEXT),
        ('Label', LABEL)
    ]
    return train_fields


def get_test_fields(TEXT):
    fields = [
        ('Context', TEXT),
        ('Ground_Truth_Utterance', TEXT),
        ('Distractor_0', TEXT),
        ('Distractor_1', TEXT),
        ('Distractor_2', TEXT),
        ('Distractor_3', TEXT),
        ('Distractor_4', TEXT),
        ('Distractor_5', TEXT),
        ('Distractor_6', TEXT),
        ('Distractor_7', TEXT),
        ('Distractor_8', TEXT)
    ]
    return fields


def get_examples(examples_path, fields, dataset_path):
    """
    :param examples_path:
    :param fields:
    :param dataset_path: dataset of generate examples
    :return: examples
    """
    if init_already(examples_path):
        examples_list = load_items(examples_path)
        examples = data.Dataset(examples=examples_list, fields=fields)
    else:
        examples = data.TabularDataset(dataset_path, format="csv",
                                      skip_header=False, fields=fields)
        dump_examples(examples, examples_path)
    return examples


def build_dict(fields, examples, vocab_path, **kwargs):
    """
    :param fields:  field to build vocab list
    :param examples:
    :param vocab_path: save_path
    :param kwargs: args pass to build_vocab(examples, **kwargs)
    :return: None
    """
    if init_already(vocab_path):
        fields.vocab = load_items(vocab_path)
    else:
        fields.build_vocab(examples, **kwargs)
        dump_vocab(fields, vocab_path)

spacy_en = spacy.load('en')
spacy_specials = ["__eot__", "__eou__"]
spacy_specials_trans = [[{ORTH:"__eot__"}], [{ORTH:"__eou__"}]]
for s, n in zip(spacy_specials, spacy_specials_trans):
    spacy_en.tokenizer.add_special_case(s, n)
def tokenizer(text):  # create a tokenizer function
    """
    __eou__
    __eot__
    :param text: preprocess
    :return: list
    """
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != '']
    tokenized_text = []
    for token in text:
        if token == "n't":
            tmp = 'not'
        elif token == "'ll":
            tmp = 'will'
        elif token == "'m":
            tmp = 'am'
        elif token == "'re":
            tmp = 'are'
        elif token == "'s":
            tmp = 'is'
        else:
            tmp = token
        if ' ' not in token: tokenized_text.append(tmp)
    return tokenized_text

def init_already(path):
    if not os.path.exists(path):
        return False
    return True


def load_items(save_path):
    with open(save_path, 'rb') as f:
        items = dill.load(f)
    return items


def dump_examples(dataset, save_path):
    with open(save_path, 'wb') as f:
        dill.dump(dataset.examples, f)


def dump_vocab(fields, save_path):
    with open(save_path, 'wb') as f:
        dill.dump(fields.vocab, f)

class Corpus(object):
    def __init__(self, dataset_path, examples_path, vocab_path, specials=[]):

        self.dataset_path = dataset_path
        self.examples_path = examples_path
        self.vocab_path = vocab_path

        self.specials = specials


    def get_iter(self, iter_type,
                 num_distractor=-1,
                 **kwargs):

        dataset_path = self.dataset_path
        examples_path = self.examples_path
        vocab_path = self.vocab_path
        specials = self.specials

        start_time = time.process_time()

        tgt_dataset_path, tgt_examples_path, tgt_vocab_path = os.path.join(dataset_path, "{}.csv".format(iter_type)), \
                                                              os.path.join(examples_path, "{}.examples".format(iter_type)), \
                                                              os.path.join(vocab_path, "{}.vocab".format(iter_type))

        TEXT = data.Field(tokenize=tokenizer,
                          include_lengths=True,
                          lower=False,
                          batch_first=True)

        if iter_type == "train":
            # LABEL = data.Field(sequential=False, is_target=True, unk_token=None)
            LABEL = data.LabelField()
            field = get_train_fields(TEXT, LABEL)
        else:
            field = get_test_fields(TEXT)

        print("generating {} examples...".format(iter_type))

        examples = get_examples(tgt_examples_path, field, tgt_dataset_path)

        step_1_time = time.process_time()
        print("done, cost time:{}".format(step_1_time - start_time))

        print("building vocabulary...")
        vectors = GloVe(name='840B',
                        dim=300,
                        cache='/remote_workspace/rs_trans/data/embeddings/')
        build_dict(TEXT, examples, tgt_vocab_path, vectors=vectors, specials=specials) #  __eot__ at [2], __eou__ at [3]
        if iter_type == "train":
            LABEL.build_vocab(examples)
        step_2_time = time.process_time()
        print("done, cost time:{}".format(step_2_time - step_1_time))
        print("building iterator...")

        iter = data.Iterator(dataset=examples, **kwargs)
        # iter = data.Iterator(dataset=examples, batch_size=16, sort=False,
        #                          train=False, repeat=False)

        step_3_time = time.process_time()
        print("done, cost time: {}".format(step_3_time - step_1_time))

        self.TEXT = TEXT

        return iter


if __name__ == "__main__":
    import argparse
    from functools import partial


    parser = argparse.ArgumentParser(description="test")

    parser.add_argument("--dataset_path", type=str, default="/remote_workspace/dataset/default/")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab")
    parser.add_argument("--examples_path", type=str, default="../data/examples")

    args = parser.parse_args()

    start_time = time.process_time()


    dataset_path, examples_path, vocab_path = partial(os.path.join, args.dataset_path), partial(os.path.join, args.examples_path), partial(os.path.join, args.vocab_path)
    train_dataset_path, test_dataset_path, valid_dataset_path = dataset_path("train.csv"), dataset_path("test.csv"), dataset_path("valid.csv")
    train_examples_path, test_examples_path, valid_examples_path = examples_path("train.examples"), examples_path("test.examples"), examples_path("valid.examples")
    train_vocab_path, test_vocab_path, valid_vocab_path = vocab_path("train.vocab"), vocab_path("test.vocab"), vocab_path("valid.vocab")

    TRAIN_TEXT, LABEL = data.Field(tokenize=tokenizer, include_lengths=True, lower=False), data.LabelField()
    TEST_TEXT, VAL_TEXT = data.Field(tokenize=tokenizer, include_lengths=True, lower=False), data.Field(tokenize=tokenizer, include_lengths=True, lower=False)

    train_fields, valid_fields, test_fields = get_train_fields(TRAIN_TEXT, LABEL), get_test_fields(VAL_TEXT), get_test_fields(TEST_TEXT)

    print("generating examples...")

    train_examples = get_examples(train_examples_path, train_fields, train_dataset_path)
    valid_examples = get_examples(valid_examples_path, valid_fields, valid_dataset_path)
    test_examples = get_examples(test_examples_path, test_fields, test_dataset_path)
    step_1_time = time.process_time()
    print("done, cost time:{}".format(step_1_time - start_time))

    print("building vocabulary...")

    build_dict(TRAIN_TEXT, train_examples, train_vocab_path, specials=['__eot__', '__eou__']) #  __eot__ in [2]
    build_dict(VAL_TEXT, valid_examples, valid_vocab_path, specials=['__eot__', '__eou__']) #  __eot__ in [2]
    build_dict(TEST_TEXT, test_examples, test_vocab_path, specials=['__eot__', '__eou__']) #  __eot__ in [2]
    step_2_time = time.process_time()
    print("done, cost time:{}".format(step_2_time - step_1_time))

    print("building iterator...")

