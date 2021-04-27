import torch
import torch.nn


# modify by https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def get_mask_from_seq_lens(seq_lens): # TODO => 4 dim
    """

    :param seq_lens: batch of seq_lens, shape (b, s)
    :return: bool
    """
    mask_ons = seq_lens.new_ones(seq_lens.size(0), torch.max(seq_lens))
    return seq_lens.unsqueeze(1) >= mask_ons.cumsum(dim=1)


def padding_turn(turn_seq, padding_seq_len, padding_turn_len): # [[[]]], b, t, s  O(b * t * s)
    new_c = []
    for turn in turn_seq:
        new_turn = []
        if len(turn) <= padding_turn_len:
            turn.extend([[]] * (padding_turn_len - len(turn)))
        for seq in turn:
            if len(seq) <= padding_seq_len:
                seq.extend([1] * (padding_seq_len - len(seq)))
                new_turn.append(seq)
        new_c.append(new_turn)
    return new_c


def split_turn(c): # c shape(b, s)
    new_bc, _batch_turn_len, _padding_seq_len, _padding_turn_len = [], [], -1, -1
    for bc in c:
        tmp_seq = []  # seq in turn
        tmp_turn = []  # t
        _seq_len_in_turn = []  # seq len in turn

        for word in bc:
            if word.item() != 2 and word.item() != 1:
                tmp_seq.append(word.item())
            else:
                if not tmp_seq:
                    break
                tmp_turn.append(tmp_seq)
                _seq_len_in_turn.append(len(tmp_seq))
                _padding_seq_len = max(len(tmp_seq), _padding_seq_len)
                tmp_seq = []
        new_bc.append(tmp_turn)
        _batch_turn_len.append(_seq_len_in_turn)
        _padding_turn_len = max(len(_seq_len_in_turn), _padding_turn_len)
    return new_bc, _batch_turn_len

# from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def masked_softmax(vector, mask):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (
            result.sum(dim=-1, keepdim=True) + 1e-13
        )
    return result


def weighted_sum_multi(vector, attn_weight): # TODO => over turn
    weighted_sum = attn_weight.bmm(vector)

    return weighted_sum

def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)
    if mask == None: return weighted_sum

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous()

    return weighted_sum * mask

def masked_max(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    if not mask: return torch.max(vector, dim=dim, keepdim=keepdim)[0]
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    if not mask: return torch.mean(vector, dim=dim, keepdim=keepdim)
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=tiny_value_of_dtype(vector.dtype))

# https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L1984
def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)

def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
