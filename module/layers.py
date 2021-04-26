import torch.nn as nn
from utils import *

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx.cuda())

        return reordered_outputs

class RNN_encoder(nn.Module):
    """
    pack the padded sequences
    """
    def __init__(self,
                 rnn,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 num_layers=1,
                 **kwargs):
        super(RNN_encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self._encoder = rnn(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            **kwargs)


    def forward(self, inp, inp_len):
        if not inp_len: return self._encoder(inp)
        packed_batch = nn.utils.rnn.pack_padded_sequence(inp,
                                                         inp_len,
                                                         batch_first=self.batch_first,
                                                         enforce_sorted=False)
        outputs, output_c = self._encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        return outputs, output_c

# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
class RNNDropout(nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):

        """
        Apply dropout to input tensor.
        # Parameters
        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`
      a  # Returns
        output : `torch.FloatTensor`
    f        A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class SublayerConnection_PreLN(nn.Module):
    """
    residual connection with pre-ln
    """

    def __init__(self, d_model, dropout, cross=False):
        super(SublayerConnection_PreLN, self).__init__()
        self.cross = cross
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if cross:
            self.norm_bak = nn.LayerNorm(d_model)

    def forward(self, x, sublayer, y=None):
        if self.cross:
            return x + self.dropout(sublayer(self.norm(x), self.norm_bak(y), self.norm_bak(y), need_weights=False)[0])
        return x + self.dropout(sublayer(self.norm(x)))

class SublayerConnection_PostLN(nn.Module):
    """
    residual connection with post-ln
    """

    def __init__(self, d_model, dropout, cross=False):
        super(SublayerConnection_PostLN, self).__init__()
        self.cross = cross
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, y=None):
        if self.cross:
            return self.norm(x + self.dropout(sublayer(x, y, y, need_weights=False)[0]))
        return self.norm(x + self.dropout(sublayer(x)))


class SoftmaxAttention(nn.Module):
    # input shape should be (b, seq_len, dim)
    def forward(self,
                context,
                response,
                context_mask=None,
                response_mask=None):
        similarity_matrix = context.bmm(response.transpose(2, 1).contiguous())

        c_attn = masked_softmax(similarity_matrix, response_mask)
        r_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), context_mask)

        attend_c = weighted_sum(response, c_attn, context_mask)
        attend_r = weighted_sum(context, r_attn, response_mask)
        # attend_c = c_attn.bmm(response)
        # attend_r = r_attn.bmm(context)

        return attend_c, attend_r


class DotAttention(nn.Module):

    def forward(self,
                q, k, v,
                v_mask=None,
                dropout=None):
        similarity_matrix = q.bmm(k.transpose(-1, -2).contiguous())

        attend_v = masked_softmax(similarity_matrix, v_mask)
        if dropout:
            attend_v = dropout(attend_v)
        output = attend_v.bmm(v)
        return output # (b, m, d)



# https://github.com/allenai/allennlp/blob/master/allennlp/modules/masked_layer_norm.py
class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.
    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float = 0.1) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size

    def forward(self, tensor: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        broadcast_mask = mask.unsqueeze(-1)
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt(
            (masked_centered * masked_centered).sum() / num_elements
            + tiny_value_of_dtype(tensor.dtype)
        )
        return (
            self.gamma * (tensor - mean) / (std + tiny_value_of_dtype(tensor.dtype))
            + self.beta
        )