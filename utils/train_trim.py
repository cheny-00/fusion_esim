
import dill
import os 
import numpy as np
import pickle
import tqdm
path = '/home/chy/remote_workspace/dataset/default/valid.csv'
train_path = '/home/chy/remote_workspace/dataset/default/train.csv'
save_path = '/home/chy/remote_workspace/fusion_esim/data/w2v'
model_name = 'bert'
bert_path = '/home/chy/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-12_H-768_A-12'

# train_dataset = UbuntuCorpus(path=train_path, type='train', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
# val_dataset = UbuntuCorpus(path=path, type='valid', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)
# test_dataset = UbuntuCorpus(path=path, type='test', save_path=save_path, model_name=model_name, special=['__eou__', '__eot__'], bert_path=bert_path)



def _padding(text, max_len):
    if len(text) < max_len:
        text = np.append(text, np.zeros(max_len - len(text)))
    return text

def padding(text, max_len):
    while len(text) < max_len:
        text.append(0)
    return text 
def max_trim(text, max_len, rm_pos):
    while max_len < len(text):
        text.pop(rm_pos)
    return text
with open(os.path.join(save_path, 'train'), 'rb') as f:
    s_data =  dill.load(f)


t_con = list()
c_len = list()
t_res = list()
r_len = list()
tmp_data = s_data[0]
for line in tqdm.tqdm(tmp_data):
    # line = np.array(line)[len(line)-280:]
    line = max_trim(line, 280, 0)
    c_len.append(len(line))
    line = padding(line, 280)
    t_con.append(line)

for line in tqdm.tqdm(s_data[1]):
    # line = np.array(line)[len(line)-40:]
    line = max_trim(line, 40, -1)
    r_len.append(len(line))
    line = padding(line, 40)
    t_res.append(line)

e_data = dict()
e_data['context'] = np.array(t_con)
e_data['c_len'] = np.array(c_len)
e_data['response'] = np.array(t_res)
e_data['r_len'] = np.array(r_len)
e_data['label'] = s_data[2]

save_file = os.path.join(save_path, 'esim_data.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(e_data, f)