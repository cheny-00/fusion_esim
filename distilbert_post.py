
from utils.post_train_dataset import BertPostTrainingDataset
from utils.exp_utils import create_exp_dir, save_checkpoint

import os
from time import time, strftime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertConfig

from model.distilbert_post_train_model import *



def post_train(args):
    ##########################################################################################
    # logging
    ##########################################################################################
    work_name = args.proj_name
    work_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'logs')
    work_dir = os.path.join(work_dir, work_name)
    work_dir = os.path.join(work_dir, strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(work_dir,
                             scripts_to_save=[__file__, '../utils/create_bert_post_training_data.py',
                                              '../utils/post_train_dataset.py'],
                             debug=args.debug)

    save_dir = os.path.join(args.save_dir, work_name, strftime('%Y%m%d-%H%M%S'))
    ##########################################################################################
    ##   load post-train dataset
    ##########################################################################################
    train_dataset = BertPostTrainingDataset(args.examples_path, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    ##########################################################################################
    #   prepare train
    ##########################################################################################

    epoch_start = 0
    accu_loss = count = accumulate_batch = 0
    total_step = 0
    optim_state_dict = None
    if args.restart_file and not os.path.exists(args.restart_file):
        print("Can't find file or directory. Train will start from scratch.")
    if args.restart_file and os.path.exists(args.restart_file):
        with open(args.restart_file, 'rb') as f:
            ckpt = torch.load(f)
            epoch_start, accu_loss = ckpt['epoch'], ckpt['train_loss']
            assert epoch_start < args.epoch, 'epoch out of boundary'
            optim_state_dict = ckpt['optimizer_state_dict']
            pre_train_state = ckpt['model_state_dict']
    else:
        pre_train_state = torch.load(os.path.join(args.bert_path, 'pytorch_model.bin'),
                                     map_location='cpu')

    bert_config = DistilBertConfig.from_pretrained(args.bert_path)
    model = DistilBERTPostTrain(bert_config, pre_train_state)
    model.resize_token_embeddings(model.config.vocab_size + 2)
    del pre_train_state
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if optim_state_dict:
        optimizer.load_state_dict(optim_state_dict)
        del optim_state_dict
    ##########################################################################################
    #   Train step
    ##########################################################################################
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.cuda.set_device(0)
    model.to(device)

    log_start_time = time()
    for epoch in range(epoch_start, args.epochs):
        model.train()
        tqdm_train_dataloader = tqdm(train_dataloader)

        for idx, data in enumerate(tqdm_train_dataloader):

            msk_lm_label = data["masked_lm_labels"]
            data["masked_lm_labels"] = torch.where(msk_lm_label == -1, torch.tensor(-100), msk_lm_label)
            loss = model(input_ids=data["input_ids"].to(device),
                         token_type_ids=data["token_type_ids"].to(device),
                         attention_mask=data["attention_mask"].to(device),
                         labels=data["masked_lm_labels"].to(device),
                         next_sentence_label=data["next_sentence_labels"].to(device)).loss
            
            loss.backward()
            accu_loss += loss.item()
            count += 1

            accumulate_batch += data["next_sentence_labels"].shape[0]
            if args.virtual_batch_size == accumulate_batch or \
                    idx == (len(train_dataset) // args.batch_size): # last batch
                nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()

                total_step += 1
                elapsed, log_start_time = time() - log_start_time, time()
                log_str = '| epoch {:3d} total_step {:>8d}  | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.3f}'.format(epoch, total_step, idx + 1,optimizer.param_groups[0]['lr'],
                                                                     elapsed * 1000, accu_loss)
                logging(log_str, print_=False)
                tqdm_train_dataloader.set_description(log_str)

                accu_loss = count = accumulate_batch = 0
                if total_step % args.checkpoint_step == 0:
                    save_checkpoint(model,
                                    optimizer,
                                    save_dir,
                                    epoch,
                                    accu_loss,
                                    0,
                                    model_name='bert')





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Post train")
    parser.add_argument("--proj_name", type=str, default="bert_post_train",
                        help='project name')
    parser.add_argument("--dataset_path", type=str, default="/remote_workspace/dataset/default/",
                        help='path to dataset')
    parser.add_argument("--examples_path", type=str, default="/remote_workspace/fusion_esim/data/bert_with_eot/ubuntu_post_training.hdf5",
                        help='path to dump examples')
    parser.add_argument("--save_dir", type=str, default="../checkpoints",
                        help='checkpoints save dir')
    parser.add_argument('--restart_file', type=str, default='',
                        help='path to load checkpoint')
    parser.add_argument('--epochs', type=int, default=3,
                        help='total epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument("--bert_path", type=str, default="../data/pre_trained_ckpt/uncased_L-12_H-768_A-12",
                        help='load pretrained bert ckpt files')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='initial learning rate')
    parser.add_argument('--virtual_batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--max_gradient_norm', type=int, default=5,
                        help='clip gradient')
    parser.add_argument('--checkpoint_step', type=int, default=2500,
                        help='save checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()

    post_train(args)
