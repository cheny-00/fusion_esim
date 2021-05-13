import os
import time
import torch
import torch.nn as nn
from eval.evaluation import eval_samples
from utils import get_mask_from_seq_lens
class Trainer():

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
        self.batch_size = batch_size
        self.crit = crit
        self.fp16 = fp16
        self.logging = logging
        self.log_interval = log_interval
        self.plotter = plotter
        self.train_step = 0
        self.train_loss = 0
        self.model_name = model_name
        if self.crit == "cross_entropy":
            self.crit = nn.CrossEntropyLoss()
        elif self.crit == "mse":
            self.crit == nn.MSELoss()
        elif self.crit == "distillation":
            self.crit = nn.CrossEntropyLoss()
            self.logit_crit = nn.MSELoss()

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

    def train_process(self, *args):
        return NotImplementedError()
    def eval_process(self, *args):
        return NotImplementedError()
    def fusion_train(self,
                     epoch_num,
                     **kwargs):
        self.model.train()
        log_start_time = 0
        for batch, data in enumerate(self.train_iter):
            loss = self.train_process(data)
            if type(loss) is str: continue
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
                # if self.train_step < warmup_step:
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

    def fusion_evaluate(self):

        self.model.eval()

        total_loss, n_con, eval_start_time = 0, 0, 0
        prob_set = []

        with torch.no_grad():

            for i, data in enumerate(self.eval_iter):
                prob, n_con, total_loss = self.eval_process(data, n_con, total_loss)
                if type(prob) is str: continue
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
    @property
    def get_train_loss(self):
        return self.train_loss

class LSTMTrainer(Trainer):

    def train_process(self, data):

        label = data[2]
        if len(label) != self.batch_size: return "continue"
        elif not set(list(label)) == set([0, 1]): return "continue"
        self.optimizer.zero_grad()
        self.train_step += 1
        label = torch.tensor(label, dtype=torch.long)

        x_1_logit, x_2_logit = self.model(data)
        loss = (self.crit(x_1_logit, label.to(self.device)) +
                self.crit(x_2_logit, label.to(self.device))) / 2
        return loss

    def eval_process(self, data, n_con, total_loss):
        b_neg = data[2]
        if len(data[0][0]) != self.batch_size: return "continue"
        n_con += 1
        eva_lg_e, eva_lg_b = self.model(data)
        loss = (self.crit(eva_lg_e, torch.tensor([1] * self.batch_size).to(self.device)) +
                self.crit(eva_lg_b, torch.tensor([1] * self.batch_size).to(self.device))) / 2
        prob = nn.functional.softmax(nn.functional.softmax(eva_lg_e, dim=1)[:, 1].unsqueeze(1) +
                                     nn.functional.softmax(eva_lg_b, dim=1)[:, 1].unsqueeze(1), dim=1)
        total_loss += loss.item()
        for b_sample in b_neg:
            data = (data[0], b_sample)
            x_1_eva_f_lg, x_2_eva_f_lg = self.model(data)
            #  TODO 驗證方法 R10@1 R10@5 R2@1 MAP MMR | R@n => 是否在前n位
            loss = (self.crit(x_1_eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device)) +
                    self.crit(x_2_eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device))) / 2
            total_loss += loss.item()
            prob = torch.cat((prob, nn.functional.softmax(
                nn.functional.softmax(x_1_eva_f_lg, dim=1)[:, 1].unsqueeze(1) + \
                nn.functional.softmax(x_2_eva_f_lg, dim=1)[:, 1].unsqueeze(1), dim=1)),
                             dim=1)
        return prob, n_con, total_loss

class FineTuningTrainer(Trainer):

    def train_process(self, data):
        label = data[2]
        if len(label) != self.batch_size: return "continue"
        elif not set(list(label)) == set([0, 1]): return "continue"
        self.optimizer.zero_grad()
        self.train_step += 1
        label = torch.tensor(label, dtype=torch.long)
        x_1_logit, x_2_logit = self.model(data, fine_tuning=True)
        loss = self.crit(x_2_logit, label.to(self.device))
        return loss

    def eval_process(self, data, n_con, total_loss):
        b_neg = data[2]
        if len(data[0][0]) != self.batch_size: return "continue", n_con, total_loss
        n_con += 1
        eva_lg_e, eva_lg_b = self.model(data, fine_tuning=True)
        loss = self.crit(eva_lg_b, torch.tensor([1] * self.batch_size).to(self.device))
        prob = nn.functional.softmax(eva_lg_b, dim=1)[:, 1].unsqueeze(1)
        total_loss += loss.item()
        for b_sample in b_neg:
            data = (data[0], b_sample)
            x_1_eva_f_lg, x_2_eva_f_lg = self.model(data, fine_tuning=True)
            #  TODO 驗證方法 R10@1 R10@5 R2@1 MAP MMR | R@n => 是否在前n位
            loss = self.crit(x_2_eva_f_lg, torch.tensor([0] * self.batch_size).to(self.device))
            total_loss += loss.item()
            prob = nn.functional.softmax(x_2_eva_f_lg, dim=1)[:, 1].unsqueeze(1)
        return prob, n_con, total_loss