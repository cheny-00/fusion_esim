import os
import time
import torch
import torch.nn as nn
from eval.evaluation import eval_samples
from utils import get_mask_from_seq_lens
class Trainer:

    def __init__(self,
                 model,
                 train_iter,
                 eval_iter,
                 optimizer,
                 crit,
                 batch_size,
                 fp16,
                 logging,
                 log_interval,
                 plotter,
                 model_name
                 ):

        self.model = model
        self.device = next(model.parameters()).device
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.optimizer = optimizer
        self.crit = crit
        self.batch_size = batch_size
        self.fp16 = fp16
        self.logging = logging
        self.log_interval = log_interval
        self.plotter = plotter
        self.train_step = 0
        self.train_loss = 0
        self.model_name = model_name

        # if 'bert' in self.model_name:
        #     # TODO BertConfig Class
        #     self._proj = nn.Sequential(
        #         nn.Linear(768, 300),
        #         nn.ReLU()
        #     )
        #     self._proj.to(self.device)

        if self.fp16:
            if not 'cuda' in self.device.type:
                print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
                self.fp16 = False
            else:
                try:
                    from apex import amp
                except:
                    print('WARNING: apex not installed, ignoring --fp16 option')
                    self.fp16 = False

    def train(self,
              epoch_num,
              pre_train_model,
              **kwargs):
        self.model.train()
        device = self.device
        log_start_time = 0
        for batch, (context, response, label) in enumerate(self.train_iter):
            if len(label) != self.batch_size:
                continue
            elif not set(list(label)) == set([0, 1]):
                continue
            self.optimizer.zero_grad()
            self.train_step += 1

            if not 'bert' in self.model_name:
                context = (pre_train_model(context[0].to(device)), context[1])
                response = (pre_train_model(response[0].to(device)), response[1])

            else:
                c = pre_train_model(input_ids=context[0].to(device),
                                    attention_mask=get_mask_from_seq_lens(context[1].to(device))).last_hidden_state
                r = pre_train_model(input_ids=response[0].to(device),
                                    attention_mask=get_mask_from_seq_lens(response[1].to(device))).last_hidden_state
                context, response = (c, context[1]), (r, response[1])


            logit = self.model(context[0].to(device),
                               context[1].cpu(),
                               response[0].to(device),
                               response[1].cpu())

            loss = self.crit(logit, label.to(device))

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 10.0)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            if 'scheduler' and 'warmup_step' in kwargs:
                scheduler = kwargs['scheduler']
                warmup_step = kwargs['warmup_step']
                if self.train_step < warmup_step:
                    scheduler.step()
            self.train_loss += loss.float().item()

            if self.train_step % self.log_interval == 0:
                cur_loss = self.train_loss / self.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {}'.format(epoch_num, self.train_step, batch+1, self.optimizer.param_groups[0]['lr'],
                                                                elapsed * 1000 / self.log_interval, cur_loss)
                self.logging(log_str)
                self.plotter.plot('train_loss', 'train', 'loss curve', self.train_step, cur_loss)
                self.train_loss = 0
                log_start_time = time.time()

    def evaluate(self,
                 pre_train_model):

        self.model.eval()
        pre_train_model.eval()

        total_loss, n_con, eval_start_time = 0, 0, 0
        prob_set = []

        with torch.no_grad():

            for i, (c, s, neg) in enumerate(self.eval_iter):
                if pre_train_model:
                    context, context_len = pre_train_model(c[0].to(self.device)), c[1]
                    s = (pre_train_model(s[0].to(self.device)), s[1])
                else:
                    context, context_len = c[0], c[1]
                if context.size(0) != self.batch_size:
                    continue
                n_con += 1
                eva_lg = self.model(context.to(self.device), context_len.cpu(), s[0].to(self.device), s[1].cpu())
                loss = self.crit(eva_lg, torch.tensor([1] * self.batch_size).to(self.device))
                prob = nn.functional.softmax(eva_lg, dim=1)[:, 1].unsqueeze(1)
                total_loss += loss.item()
                for sample in neg:
                    if pre_train_model:
                        sample = (pre_train_model(sample[0].to(self.device)), sample[1])
                    eva_f_lg = self.model(context, context_len.cpu(), sample[0], sample[1].cpu())
                    #  TODO 驗證方法 R10@1 R10@5 R2@1 MAP MMR | R@n => 是否在前n位
                    loss = self.crit(eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device))
                    total_loss += loss.item()
                    prob = torch.cat((prob, nn.functional.softmax(eva_f_lg, dim=1)[:, 1].unsqueeze(1)), dim=1)
                prob_set.append(prob.tolist())
            eva = eval_samples(prob_set)
        self.logging('-' * 100)
        log_str = '| Eval at step {:>8d} | time: {:5.2f}s ' \
                  '| valid loss {:5.4f} | R_1@2 {:5.4f} | R_1@10 {:5.4f} | R_2@10 {:5.4f} |'\
                  ' R_5@10 {:5.4f} | MAP {:5.4f} | MRR {:5.4f}  '.format(self.train_step,
                                                                         (time.time() - eval_start_time),
                                                                         total_loss / (n_con * 10),
                                                                         eva[0], eva[1], eva[2], eva[3], eva[4], eva[5])
        self.logging(log_str)
        self.logging('-' * 100)
        self.model.train()
        pre_train_model.train()
        return eva, (total_loss / (n_con * 10))

    def fusion_train(self,
                     epoch_num,
                     **kwargs):
        self.model.train()
        device = self.device
        log_start_time = 0
        for batch, data in enumerate(self.train_iter):
            w2v_data, b_data, label = data[0], data[1], data[0][2]
            if len(label) != self.batch_size: continue
            elif not set(list(label)) == set([0, 1]): continue
            self.optimizer.zero_grad()
            self.train_step += 1
            label = torch.tensor(label, dtype=torch.long)

            x_1_logit, x_2_logit = self.model(w2v_data,
                                              b_data)
            loss = self.crit(x_1_logit, label.to(device))  #+ \
                #    self.crit(x_2_logit, label.to(device))

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 10.0)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            # if 'scheduler' and 'warmup_step' in kwargs:
            #     scheduler = kwargs['scheduler']
            #     warmup_step = kwargs['warmup_step']
            #     if self.train_step < warmup_step:
            #         scheduler.step()
            self.train_loss += loss.float().item()

            if self.train_step % self.log_interval == 0:
                cur_loss = self.train_loss / self.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {}'.format(epoch_num, self.train_step, batch+1, self.optimizer.param_groups[0]['lr'],
                                                                elapsed * 1000 / self.log_interval, cur_loss)
                self.logging(log_str)
                self.plotter.plot('train_loss', 'train', 'loss curve', self.train_step, cur_loss)
                self.train_loss = 0
                log_start_time = time.time()

    def fusion_evaluate(self):

        self.model.eval()

        total_loss, n_con, eval_start_time = 0, 0, 0
        prob_set = []

        with torch.no_grad():

            for i, data in enumerate(self.eval_iter):
                w2v_data, b_data = data[0], data[1]
                w_neg, b_neg = w2v_data[2], b_data[2]
                if len(w2v_data[0][0]) != self.batch_size: continue
                n_con += 1
                eva_lg_e, eva_lg_b = self.model(w2v_data, b_data)
                loss = self.crit(eva_lg_e, torch.tensor([1] * self.batch_size).to(self.device))
                prob = nn.functional.softmax(eva_lg_e, dim=1)[:, 1].unsqueeze(1)
                total_loss += loss.item()
                for w_sample, b_sample in zip(w_neg, b_neg):
                    w2v_data, b_data = (w2v_data[0], w_sample), (b_data[0], b_sample)
                    x_1_eva_f_lg, x_2_eva_f_lg = self.model(w2v_data, b_data)
                    #  TODO 驗證方法 R10@1 R10@5 R2@1 MAP MMR | R@n => 是否在前n位
                    loss = self.crit(x_1_eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device)) # + \
                           # self.crit(x_2_eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device))
                    total_loss += loss.item()
                    prob = torch.cat((prob, nn.functional.softmax(x_1_eva_f_lg, dim=1)[:, 1].unsqueeze(1)), dim=1)
                prob_set.append(prob.tolist())
            eva = eval_samples(prob_set)
        self.logging('-' * 100)
        log_str = '| Eval at step {:>8d} | time: {:5.2f}s ' \
                  '| valid loss {:5.4f} | R_1@2 {:5.4f} | R_1@10 {:5.4f} | R_2@10 {:5.4f} |'\
                  ' R_5@10 {:5.4f} | MAP {:5.4f} | MRR {:5.4f}  '.format(self.train_step,
                                                                         (time.time() - eval_start_time),
                                                                         total_loss / (n_con * 10),
                                                                         eva[0], eva[1], eva[2], eva[3], eva[4], eva[5])
        self.logging(log_str)
        self.logging('-' * 100)
        self.model.train()
        return eva, (total_loss / (n_con * 10))
    def get_train_loss(self):
        return self.train_loss

