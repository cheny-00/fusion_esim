import time
from utils.data import *
from model import *
from transformers import (
    BertModel,
    DistilBertModel,
    BertConfig,
    DistilBertConfig,
)
from eval.evaluation import eval_samples


def main(args):

    test_dataset = UbuntuCorpus(path=os.path.join(args.dataset_path, 'test.csv'),
                                type='valid',
                                save_path=args.examples_path,
                                model_name=model_name,
                                special=['__eou__', '__eot__'],
                                worddict=train_dataset.Vocab.worddict,
                                bert_path=args.bert_path)
    test_iter = DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           collate_fn=ub_corpus_test_collate_fn,
                           num_workers=8,
                           drop_last=True,
                           shuffle=False)
    model = prepare_model(args.model, args.checkpoint_dir, args.bert_path)
    model.eval()
    total_loss, n_con, eval_start_time = 0, 0, 0
    prob_set = []

    with torch.no_grad():

        for i, data in enumerate(test_iter):
            prob, n_con, total_loss = eval_process(data, n_con, total_loss)
            if type(prob) is str: continue
            prob_set.append(prob.tolist())
        eva = eval_samples(prob_set)
    print('-' * 100)
    log_str = '| time: {:5.2f}s ' \
              '| valid loss {:5.4f} | R_1@2 {:5.4f} | R_1@10 {:5.4f} | R_2@10 {:5.4f} |'\
              ' R_5@10 {:5.4f} | MAP {:5.4f} | MRR {:5.4f}  '.format((time.time() - eval_start_time),
                                                                     total_loss / (n_con * 10),
                                                                     eva[0], eva[1], eva[2], eva[3], eva[4], eva[5])
    print(log_str)
    print('-' * 100)

def prepare_model(model_name, checkpoint_dir):
    if 'bert' in model_name:
        bert_config = BertConfig.from_json_file(os.path.join(args.bert_path,
                                                             'config.json'))
        pre_trianed_state = torch.load(os.path.join(args.bert_path, 'pytorch_model.bin'),
                                       map_location='cpu')
        model = BertModel.from_pretrained(args.bert_path, state_dict=pre_trianed_state)
        del pre_trianed_state
    return model

def load_model(checkpoint_dir):
    pass

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
        prob = torch.cat((prob, nn.functional.softmax(x_2_eva_f_lg, dim=1)[:, 1].unsqueeze(1)), dim=1)
    return prob, n_con, total_loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="testing")

    parser.add_argument("--dataset_path", type=str, default="/remote_workspace/dataset/default/",
                        help='path to dataset')
    parser.add_argument("--examples_path", type=str, default="/remote_workspace/rs_trans/data/bert",
                        help='path to dump examples')
    parser.add_argument("--model", type=str, default="fusion_esim",
                        help='model')
    parser.add_argument("--bert_path", type=str, default="../data/pre_trained_ckpt/uncased_L-8_H-512_A-8",
                        help='load pretrained bert ckpt files')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='checkpoint dir')
    args = parser.parse_args()
    main(args)