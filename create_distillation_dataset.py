

import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from time import time, strftime
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
from collections import defaultdict, OrderedDict

from model.fusion_esim import Bert, FusionEsim
from utils.data import UbuntuCorpus
from eval.evaluation import eval_samples
from utils.exp_utils import create_exp_dir

DistillationTrainData = dict()

DistillationTrainData['context'] = np.empty((0, 280), int)
DistillationTrainData['c_len'] = np.empty((0, 1), int)
DistillationTrainData['response'] = np.empty((0, 40), int)
DistillationTrainData['r_len'] = np.empty((0, 1), int)
DistillationTrainData['label'] = np.empty((0, 1), int)
DistillationTrainData['t_logits'] = np.empty((0, 2), int)


def post_train(args):
    ##########################################################################################
    # logging
    ##########################################################################################
    work_name = args.proj_name
    work_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'logs')
    work_dir = os.path.join(work_dir, work_name)
    work_dir = os.path.join(work_dir, strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(work_dir,
                             scripts_to_save=[__file__],
                             debug=args.debug)
    if not args.debug:
        save_dir = os.path.join(args.save_dir, work_name, strftime('%Y%m%d-%H%M%S'))
    ##########################################################################################
    ##   load post-train dataset
    ##########################################################################################
    train_dataset = UbuntuCorpus(path=os.path.join(args.dataset_path, 'train.csv'),
                                 type='train',
                                 save_path=args.examples_path,
                                 model_name=args.model_name,
                                 special=['__eou__', '__eot__'],
                                 bert_path=args.bert_path)
    train_iter = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            drop_last=True,
                            shuffle=False)
    ##########################################################################################
    #   prepare train
    ##########################################################################################

    assert os.path.exists(args.checkpoint_file), "Can't find file or directory. Train will start from scratch."

    with open(args.checkpoint_file, 'rb') as f:
        ckpt = torch.load(f)
        pre_train_state = ckpt['model_state_dict']

    tmp = OrderedDict()
    for key in pre_train_state:
        if not 'ESIM' in key:
            tmp[key[5:]] = pre_train_state[key]

    bert_config = BertConfig.from_pretrained(args.bert_path)
    BERT = BertModel.from_pretrained(args.bert_path, config=bert_config)
    BERT.resize_token_embeddings(bert_config.vocab_size + 2)

    model = Bert(BERT, dropout=0)
    model.load_state_dict(tmp)

    del pre_train_state, tmp


    ##########################################################################################
    #   Train step
    ##########################################################################################
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.cuda.set_device(0)
    model.to(device)

    model.eval()
    total_loss, eval_start_time = 0, time()
    tqdm_train_iter = tqdm(train_iter)


    with torch.no_grad():
        for i, data in enumerate(tqdm_train_iter):
            loss = eval_process(data,
                                args.batch_size,
                                device,
                                model)
            if type(loss) is str: continue
            total_loss += loss
            if (i + 1) % 200 == 0:
                log_str = '| {:>6d} batches | ms/batch {:5.2f} | loss {} |'.format(i + 1, time() - eval_start_time, total_loss / 200)
                eval_start_time, total_loss = time(), 0
                tqdm_train_iter.set_description(log_str)
                logging(log_str, print_=False)
    if not args.debug:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        with open(os.path.join(save_dir,'distillation_dataset.pkl'), 'wb') as f:
            pickle.dump(DistillationTrainData, f)
            print("saved distillation_dataset.pkl at {}".format(save_dir))


crit = nn.CrossEntropyLoss()
def eval_process(data,
                 batch_size,
                 device,
                 model):

    label = data["label"].tolist()
    if len(label) != batch_size: return "continue"
    elif not set(list(label)) == set([0, 1]): return "continue"
    label = torch.tensor(label, dtype=torch.long)
    logits = model(data["anno_seq"].to(device),
                   data["attn_mask"].to(device),
                   data["seg_ids"].to(device))
    add_to_dataset(data['esim_data'], logits)
    loss = crit(logits, label.to(device))
    return loss


def batch_bert_data(data, idx):
    batch = dict()
    for key in data:
        if key == 'esim_data': continue
        batch[key] = data[key][idx].cuda()
    return batch

def add_to_dataset(data, logits):
    DistillationTrainData['context'] = np.append(DistillationTrainData['context'], data[0][0].cpu().numpy(), axis=0)
    DistillationTrainData['c_len'] = np.append(DistillationTrainData['c_len'], data[0][1].cpu().numpy().reshape(data[0][1].shape[0], 1), axis=0)
    DistillationTrainData['response'] = np.append(DistillationTrainData['response'],data[1][0].cpu().numpy(), axis=0)
    DistillationTrainData['r_len'] = np.append(DistillationTrainData['r_len'],data[1][1].cpu().numpy().reshape(data[1][1].shape[0], 1), axis=0)
    DistillationTrainData['label'] = np.append(DistillationTrainData['label'], np.array(data[2].cpu()).reshape(data[2].shape[0], 1), axis=0)
    DistillationTrainData['t_logits'] = np.append(DistillationTrainData['t_logits'], np.array(logits.cpu()), axis=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="testing")

    parser.add_argument("--proj_name", type=str, default="evaluate bert",
                        help='project name')
    parser.add_argument("--dataset_path", type=str, default="/remote_workspace/dataset/default/",
                        help='path to dataset')
    parser.add_argument("--examples_path", type=str, default="/remote_workspace/rs_trans/data/bert_with_eot",
                        help='path to dump examples')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--checkpoint_file', type=str, default='',
                        help='checkpoint dir')

    parser.add_argument('--model_name', type=str, default='bert',
                        help='select a pre trained model')
    parser.add_argument("--bert_path", type=str, default="../data/pre_trained_ckpt/uncased_L-12_H-768_A-12",
                        help='load pretrained bert ckpt files')
    parser.add_argument("--save_dir", type=str, default="../checkpoints",
                        help='the dir to save logits')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()
    post_train(args)
