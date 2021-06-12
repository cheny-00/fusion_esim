import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.train import LSTMTrainer, FineTuningTrainer
from model import *
from utils.exp_utils import create_exp_dir, save_checkpoint
from utils.visdom_plot import VisdomLinePlotter
from utils.data import *
from utils.utils import get_mask_from_seq_lens
from transformers import (
    BertModel,
    DistilBertModel,
    BertConfig,
    DistilBertConfig,
)
from transformers.optimization import get_linear_schedule_with_warmup

def main(args):

    ###############################################################################
    # Setting
    ##############################################################################

    work_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'logs')
    work_dir = os.path.join(work_dir, args.proj_name,time.strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(work_dir,
                             scripts_to_save=['../fusion_run.py', '../utils/train.py'
                                 , '../model/{}.py'.format(args.model.lower())],
                             debug=args.debug)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    plotter = VisdomLinePlotter(env_name="{}".format(args.proj_name))
    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.cuda.set_device(0)

    ###############################################################################
    # Load dataset & pre-train model
    ##############################################################################
    # model_name = 'bert-base-uncased'
    model_name = args.model_name
    train_dataset = UbuntuCorpus(path=os.path.join(args.dataset_path, 'train.csv'),
                                 type='train',
                                 save_path=args.examples_path,
                                 model_name=model_name,
                                 special=['__eou__', '__eot__'],
                                 bert_path=args.bert_path)
    eval_dataset = UbuntuCorpus(path=os.path.join(args.dataset_path, 'valid.csv'),
                                type='valid',
                                save_path=args.examples_path,
                                model_name=model_name,
                                special=['__eou__', '__eot__'],
                                worddict=train_dataset.Vocab.worddict,
                                bert_path=args.bert_path)
    train_iter = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            drop_last=True,
                            shuffle=True
                            )
    eval_iter = DataLoader(eval_dataset,
                           batch_size=args.batch_size,
                           drop_last=True,
                           shuffle=False)


    ## Build Bert model
    if 'bert' == model_name:
        bert_config = BertConfig.from_json_file(os.path.join(args.bert_path,
                                                             'config.json'))
        ModelClass = BertModel
    elif 'distilbert' == model_name:
        ModelClass = DistilBertModel
        bert_config = DistilBertConfig.from_json_file(os.path.join(args.bert_path,
                                                                   'config.json'))
    if args.load_post_trained_bert:
        pre_trianed_state  = torch.load(args.load_post_trained_bert, map_location='cpu')['model_state_dict']
        bert_config.vocab_size += 2
        print("loaded state dict from: {}".format(args.load_post_trained_bert))
    else:
        pre_trianed_state = torch.load(os.path.join(args.bert_path, 'pytorch_model.bin'),
                                       map_location='cpu')
    bert = ModelClass.from_pretrained(args.bert_path, config=bert_config, state_dict=pre_trianed_state)
    del pre_trianed_state

    ## build word embedding
    _word_embedding = bert.embeddings.word_embeddings
    _word_embedding = train_dataset.Vocab.build_embed_layer(embedding_weight=_word_embedding.weight)
    embedding_layer = nn.Embedding.from_pretrained(_word_embedding, freeze=False, padding_idx=0)
    ###############################################################################
    # Build the model
    ###############################################################################

    if args.model == 'fusion_esim':
        model = fusion_esim.FusionEsim(BERT=bert,
                                       n_bert_token=train_dataset.Vocab.n_bert_token,
                                       n_token=-1,
                                       input_size=args.d_embed,
                                       hidden_size=args.d_model,
                                       dropout=args.dropout,
                                       dropatt=args.dropatt,
                                       embedding_layer=embedding_layer,
                                       n_layer=args.n_layer)

    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    #### scheduler
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         args.epochs // 4,
                                                         eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='max',
                                                         factor=args.decay_rate,
                                                         patience=args.patience,
                                                         min_lr=args.lr_min)
    elif args.scheduler == 'cosine_warm_up':
        total_step = len(train_dataset) // args.batch_size * max(5, args.epochs)
        warmup_step = total_step // 10
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_step,
                                                    num_training_steps=total_step)
    elif args.scheduler == 'constant':
        pass

    model.to(device)
    if args.cuda and args.fp16:
        # check https://nvidia.github.io/apex/amp.html for more detail
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    ###############################################################################
    # Train & validate
    ###############################################################################
    best_score, ckpt_epoch, train_loss = 0, 1, 0
    if args.restart:
        with open(args.restart_dir, 'rb') as f:
            ckpt = torch.load(f)
            ckpt_epoch, best_score, train_loss = ckpt['epoch'], ckpt['best_score'], ckpt['train_loss']
            assert ckpt_epoch < args.epochs, 'out of boundary'
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            model.load_state_dict(ckpt['model_state_dict'])
        if not args.fp16:
            model = model.float()
    crit = args.crit
    if args.fine_tuning: TrainerClass = FineTuningTrainer
    else: TrainerClass = LSTMTrainer
    train_model = TrainerClass(model=model,
                               train_iter=train_iter,
                               eval_iter=eval_iter,
                               optimizer=optimizer,
                               crit=crit,
                               batch_size=args.batch_size,
                               fp16=args.fp16,
                               logging=logging,
                               log_interval=args.log_interval,
                               plotter=plotter,
                               model_name=model_name,
                               train_loss=train_loss,
                               distill_loss_fn=args.distill_loss_fn,
                               temperature=args.temperature)

    save_dir = os.path.join(args.save_dir, args.proj_name, time.strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(ckpt_epoch, args.epochs + 1):
        train_model.fusion_train(epoch,
                                 scheduler=scheduler,
                                 warmup_step=args.warmup_step)
        if epoch % args.eval_interval == 0 or epoch == args.epochs or args.fine_tuning:
            eva, eval_loss = train_model.fusion_evaluate()

            # save best
            if not best_score:
                best_score = eva
            elif eva[1] > best_score[1] :
                best_score = eva
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_model.get_train_loss},
                    os.path.join(save_dir, "best.pth.tar"))
            if args.scheduler == 'dev_perf':
                scheduler.step(eva[1])
        if not epoch % 10 or args.fine_tuning:
            save_checkpoint(model,
                            optimizer,
                            save_dir,
                            epoch,
                            train_model.get_train_loss,
                            best_score)
        if not args.scheduler == 'dev_perf':
            scheduler.step()




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="training")

    parser.add_argument("--dataset_path", type=str, default="/remote_workspace/dataset/default/",
                        help='path to dataset')
    parser.add_argument("--examples_path", type=str, default="/remote_workspace/rs_trans/data/bert",
                        help='path to dump examples')
    parser.add_argument("--save_dir", type=str, default="../checkpoints",
                        help='checkpoints save dir')
    parser.add_argument("--bert_path", type=str, default="../data/pre_trained_ckpt/uncased_L-8_H-512_A-8",
                        help='load pretrained bert ckpt files')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=64,
                        help='total epochs')

    parser.add_argument('--embed_type', type=int, default=1,
                        help='embedding type')
    parser.add_argument("--model", type=str, default="fusion_esim",
                        help='model')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--pre_ln', action='store_true',
                        help='use pre-layernorm')

    parser.add_argument('--fine_tuning', action='store_true',
                        help='fine-tuning step')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--crit', default='cross_entropy', type=str,
                        choices=['cross_entropy', 'mse', 'distillation'],
                        help='loss function to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'cosine_warm_up'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='upper epoch limit')

    parser.add_argument('--seed', type=int, default=9981,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='evaluation interval')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')

    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')

    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--fp16', action='store_true',
                        help='run in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--opt_level', type=str, default='O0',
                        help='Nvidia amp opt_level')

    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument'
                             ' supersedes --static-loss-scale.')
    parser.add_argument('--model_name', type=str, default='en_core_web_sm',
                        help='select a pre trained model')
    parser.add_argument('--load_post_trained_bert', type=str, default="",
                        help="load post-trained BERT")
    parser.add_argument('--distill_loss_fn', type=str, default="mse",
                        help="loss function for distillation")
    parser.add_argument('--temperature', type=str, default=1,
                        help="distillation temperature")

    parser.add_argument('--proj_name', type=str, default='ubuntu_corpus',
                        help='project name')
    args = parser.parse_args()
    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
            args.fp16 = False
        else:
            try:
                from apex import amp
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                args.fp16 = False
    main(args)
