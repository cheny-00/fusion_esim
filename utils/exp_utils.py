import functools
import os, shutil

import numpy as np

import torch


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model,
                    optimizer,
                    save_dir,
                    epoch,
                    train_loss,
                    best_score):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "best_score": best_score,
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss},
        os.path.join(save_dir, "ESIM_like_{}.pth.tar".format(epoch)))
