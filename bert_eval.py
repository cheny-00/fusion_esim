

import os
import torch
import pickle
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

DistillationTrainData = defaultdict(list)


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
    test_dataset = UbuntuCorpus(path=os.path.join(args.dataset_path, 'test.csv'),
                                type='test',
                                save_path=args.examples_path,
                                model_name=args.model_name,
                                special=['__eou__', '__eot__'],
                                bert_path=args.bert_path)
    test_iter = DataLoader(test_dataset,
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
    total_loss, n_con, eval_start_time = 0, 0, time()
    prob_set = []

    tqdm_test_iter = tqdm(test_iter)

    total_steps = len(test_iter)
    curr = eval_interval = total_steps // args.part
    prev = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm_test_iter):
            prob, n_con, total_loss = eval_process(data,
                                                   n_con,
                                                   total_loss,
                                                   args.batch_size,
                                                   device,
                                                   model)
            if type(prob) is str: continue
            prob_set.append(prob.tolist())
            if  n_con + 1 == curr:
                log_str = get_score('Eval Step:{}'.format(curr // eval_interval),
                                    logging,
                                    prob_set[prev:curr],
                                    eval_start_time,
                                    total_loss/ (n_con * 10))
                tqdm_test_iter.set_description(log_str)
                eval_start_time = time()
                prev, curr = n_con, curr + eval_interval


    log_str = get_score('Final Eval',
                        logging,
                        prob_set,
                        eval_start_time,
                        total_loss/ (n_con * 10),)

    tqdm_test_iter.set_description(log_str)
    if not args.debug:
        with open(os.path.join(save_dir,'distillation_dataset.pkl'), 'wb') as f:
            pickle.dump(DistillationTrainData, f)
            print("saved distillation_dataset.pkl at {}".format(save_dir))

def eval_process(data,
                 n_con,
                 total_loss,
                 batch_size,
                 device,
                 model):
    crit = nn.CrossEntropyLoss()
    b_neg = data['esim_data'][2]
    if len(data['esim_data'][0][0]) != batch_size: return "continue", n_con, total_loss
    n_con += 1
    eva_lg_b = model(data["anno_seq"][0].to(device),
                     data["attn_mask"][0].to(device),
                     data["seg_ids"][0].to(device))
    add_to_dataset(data['esim_data'], eva_lg_b)
    loss = crit(eva_lg_b, torch.tensor([1] * batch_size).to(device))
    prob = nn.functional.softmax(eva_lg_b, dim=1)[:, 1].unsqueeze(1)
    total_loss += loss.item()
    for idx, b_sample in enumerate(b_neg):
        x_2_eva_f_lg = model(data["anno_seq"][idx + 1].to(device),
                             data["attn_mask"][idx + 1].to(device),
                             data["seg_ids"][idx + 1].to(device),
                             isdistilbert=False)
        #  TODO 驗證方法 R10@1 R10@5 R2@1 MAP MMR | R@n => 是否在前n位
        loss = crit(x_2_eva_f_lg, torch.tensor([0] * batch_size).to(device))
        total_loss += loss.item()
        prob = torch.cat((prob, nn.functional.softmax(x_2_eva_f_lg, dim=1)[:, 1].unsqueeze(1)), dim=1)
    return prob, n_con, total_loss

def get_score(n_eval,
              logging,
              prob_set,
              eval_start_time,
              curr_loss):

    eva = eval_samples(prob_set)
    logging('-' * 100)
    log_str = '|{}| time: {:5.2f}s ' \
              '| valid loss {:5.4f} | R_1@2 {:5.4f} | R_1@10 {:5.4f} | R_2@10 {:5.4f} |'\
              ' R_5@10 {:5.4f} | MAP {:5.4f} | MRR {:5.4f}  '.format(n_eval,
                                                                     (time() - eval_start_time),
                                                                     curr_loss,
                                                                     eva[0], eva[1], eva[2], eva[3], eva[4], eva[5])
    logging(log_str)
    logging('-' * 100)
    return log_str

def batch_bert_data(data, idx):
    batch = dict()
    for key in data:
        if key == 'esim_data': continue
        batch[key] = data[key][idx].cuda()
    return batch

def add_to_dataset(data, logits):
    DistillationTrainData['esim_data'].append(data)
    DistillationTrainData['t_logits'].append(logits)

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

    parser.add_argument('--part', type=int, default=10,
                        help='eval part')
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
