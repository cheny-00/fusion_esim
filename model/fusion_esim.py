import functools
import torch
import torch.nn as nn
from utils import get_mask_from_seq_lens, masked_max, masked_mean
from utils import utils_esim
from module.layers import RNN_encoder, RNNDropout, SoftmaxAttention, MaskedLayerNorm, Seq2SeqEncoder
from transformers import BertModel

class ESIM_like(nn.Module):
    def __init__(self,
                 n_token,
                 input_size,
                 hidden_size,
                 dropout,
                 dropatt,
                 n_layer,
                 bert_embeddings,
                 embedding_layer,
                 ismasked=True):
        super(ESIM_like, self).__init__()
        # hyper parameters
        self.input_size = bert_embeddings.embedding_dim
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.dropatt = dropatt
        self.drop = nn.Dropout(dropout)
        self.d_model = self.hidden_size
        self.ismasked = ismasked

        self.rnn_drop = RNNDropout(p=self.dropout)
        # word emb
        self.bert_embeddings = bert_embeddings
        self.embedding = embedding_layer

        # word level
        self.token_enc = RNN_encoder(nn.LSTM,
                                     input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=1,
                                     bidirectional=True,
                                     bias=True,
                                     dropout=0.0)

        self.softmax_attention = SoftmaxAttention()
        self.projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size,
                                                  self.hidden_size),
                                        nn.ReLU())
        self.composition = RNN_encoder(nn.LSTM,
                                       input_size=self.d_model,
                                       hidden_size=self.d_model,
                                       num_layers=1,
                                       bidirectional=True,
                                       bias=True,
                                       dropout=0.0)

        # classifier
        self._classifier = nn.Sequential(self.drop,
                                         nn.Linear(2 * 4 * self.hidden_size, self.hidden_size),
                                         nn.Tanh(),
                                         self.drop,
                                         nn.Linear(self.hidden_size, 2)
                                         )
        if self.ismasked:
            self.emb_ln_c = nn.LayerNorm(self.input_size)
            self.emb_ln_r = nn.LayerNorm(self.input_size)
            self.com_ln_c = nn.LayerNorm(self.d_model)
            self.com_ln_r = nn.LayerNorm(self.d_model)
        else:
            self.emb_ln_c = MaskedLayerNorm(self.input_size)
            self.emb_ln_r = MaskedLayerNorm(self.input_size)
            self.com_ln_c = MaskedLayerNorm(self.d_model)
            self.com_ln_r = MaskedLayerNorm(self.d_model)
        # Initialize all weights and biases in the model.
        self.apply(_init_weights)

    def forward(self, w2v_data, b_data):
        # inp = tuple(zip(*(w2v_data, b_data))) # c, c_l, r, r_l
        context_len, response_len = None, None
        context_embed, response_embed = tuple(map(lambda x:self.bert_embeddings(x[0].clone().detach().cuda()), b_data[:2]))

        # word embed
        if not self.ismasked:
            c_mask = get_mask_from_seq_lens(context_len).cuda()
            r_mask = get_mask_from_seq_lens(response_len).cuda()
        else:
            c_mask, r_mask = None, None

        # context_embed, response_embed = self.embedding(context.clone().detach()), self.embedding(response.clone().detach())
        # context_embed, response_embed = self.emb_ln_c(context), self.emb_ln_r(response)

        # word level
        # out_c_w, _ = self.token_enc(context_embed, context_len)
        # out_r_w, _ = self.token_enc(response_embed, response_len)
        context_embed, response_embed = self.rnn_drop(context_embed), self.rnn_drop(response_embed)
        out_c_w, _ = self.token_enc(context_embed, context_len)
        out_r_w, _ = self.token_enc(response_embed, response_len)

        # out_c_w = self.rnn_drop(out_c_w)
        # out_r_w = self.rnn_drop(out_r_w)
        # out_c_w, out_r_w = self.trans(context_embed), self.trans(response_embed)

        # softmax attention

        attend_c, attend_r = self.softmax_attention(out_c_w, out_r_w, c_mask, r_mask)
        enhanced_c = torch.cat((out_c_w,
                                attend_c,
                                out_c_w - attend_c,
                                out_c_w * attend_c),
                               dim=-1)
        enhanced_r = torch.cat((out_r_w,
                                attend_r,
                                out_r_w - attend_r,
                                out_r_w * attend_r),
                               dim=-1)
        # (b, s, d)
        projected_c = self.projection(enhanced_c)
        projected_r = self.projection(enhanced_r)
        projected_c, projected_r = self.rnn_drop(projected_c), self.rnn_drop(projected_r)


        # c
        c_agg, _ = self.composition(projected_c, context_len)
        r_agg, _ = self.composition(projected_r, response_len)
        # c_agg, _ = self.composition(self.com_ln_c(projected_c), context_len)
        # r_agg, _ = self.composition(self.com_ln_r(projected_r), response_len)

        # aggregated projection (b, s, d)
        # c_mask = c_mask.unsqueeze(-1)
        c_mean = masked_mean(c_agg, c_mask, 1, True).squeeze(-2)
        r_mean = masked_mean(r_agg, r_mask, 1, True).squeeze(-2)
        c_max = masked_max(c_agg, c_mask, 1, True).squeeze(-2)
        r_max = masked_max(r_agg, r_mask, 1, True).squeeze(-2)

        aggregated = torch.cat((c_max,
                                c_mean,
                                r_max,
                                r_mean), dim=-1)
        # aggregated = torch.cat((c_max,
        #                        r_max,
        #                        c_agg[:, 1, :],
        #                        r_agg[:, 1, :]), dim=-1)
        logit = self._classifier(aggregated)
        return logit

    @staticmethod
    def __padding(x_1, x_2):
        # x_1: w2v_data, x_2 => bert_data
        if x_1.size(1) < x_2.size(1):
            x_1 = torch.cat((x_1, torch.zeros(x_1.size(0),
                                              x_2.size(1)-x_1.size(1),
                                              x_1.size(2)).cuda()), dim=1)
        return x_1[:, :x_2.size(1), :], x_2

class Bert(nn.Module):

    def __init__(self,
                 BERT,
                 n_bert_token=0,
                 dropout=0.5):
        super(Bert, self).__init__()
        self.BERT = BERT
        bert_dim = BERT.embeddings.word_embeddings.embedding_dim
        self.drop = nn.Dropout(dropout)
        self.n_bert_token = n_bert_token - 1
        self._classifier = nn.Sequential(self.drop,
                                         nn.Linear(bert_dim, bert_dim),
                                         nn.Tanh(),
                                         self.drop,
                                         nn.Linear(bert_dim, 2))

    def forward(self, c, c_len, r, r_len):
        SEP, UNK = 102, 100
        c, r = c.clone().detach().cuda(), r.clone().detach().cuda()
        c_len, r_len = c_len.cuda(), r_len.cuda()
        # c, r = c.masked_fill(c > self.n_bert_token, torch.tensor(UNK)), \
        #        r.masked_fill(r > self.n_bert_token, torch.tensor(UNK))
        c_mask = get_mask_from_seq_lens(c_len)
        convert_pad = torch.ones_like(c) * SEP * torch.logical_not(c_mask)
        c += convert_pad
        if c.size(1) + r.size(1) > 512:
            r = r[:, :257]
            d = 512 - r.size(1)
            c = c[:, :d]
            r_len.clamp_(max=257)
        input_ids = torch.cat((c, r[:, 1:]), dim=1)
        input_len = c.size(1) + r_len - 1
        attn_mask = get_mask_from_seq_lens(input_len, 511)
        token_ids = torch.cat((torch.zeros(c.size()[:2]),
                               torch.ones(r.size(0), r.size(1)-1)), dim=1).long().cuda()

        if not input_ids.size(1) == attn_mask.size(1):
            print(input_ids.size(), attn_mask.size(), input_len)
            print(c.size(), r.size(), r_len)
            raise AssertionError
       # # input_ids, attn_mask, token_ids =
        output = self.BERT(input_ids=input_ids,
                           attention_mask=attn_mask,
                           token_type_ids=token_ids).last_hidden_state

        logit = self._classifier(output[:, 0])
        return logit

class FusionEsim(nn.Module):

    def __init__(self,
                 BERT,
                 n_bert_token=0,
                 **kwargs):
        super(FusionEsim, self).__init__()
        self.ESIM = ESIM_like(bert_embeddings=BERT.embeddings.word_embeddings, **kwargs)
        self.Bert = Bert(BERT,
                         n_bert_token,
                         kwargs['dropout'])
        self.crit = nn.CrossEntropyLoss()
    def forward(self,
                br):
        esim_logit = self.ESIM(br)

        bert_logit = self.Bert(*br[0], *br[1])

        return esim_logit, bert_logit


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
    # elif isinstance(module, nn.Embedding):
    #     nn.init.xavier_uniform_(module.weight)
