

# learn from https://github.com/baidu/Dialogue/blob/master/DAM/utils/evaluation.py
def p_at_n_m(inp, pos_score, n, m):

    curr = inp[:m]
    curr = sorted(curr, reverse=True)
    if curr[n-1] <= pos_score:
        return 1
    return 0


def average_precision(inp, pos_score):
    return pos_score / (inp.index(pos_score) + 1)


def reciprocal_rank(inp, pos_score):
    return 1 / (inp.index(pos_score) + 1)


def accuracy():
    pass


def eval_samples(inp): # inp [[]]: shape [batch], batch: (16, 10)
    length = 0

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0
    ap, rr = 0.0, 0.0

    for b_set in inp:
        for batch in b_set:
            pos_score = batch[0]
            assert len(batch) == 10

            sorted_batch = sorted(batch, reverse=True)

            p_at_1_in_2 += p_at_n_m(batch, pos_score, 1, 2)
            p_at_1_in_10 += p_at_n_m(batch, pos_score, 1, 10)
            p_at_2_in_10 += p_at_n_m(batch, pos_score, 2, 10)
            p_at_5_in_10 += p_at_n_m(batch, pos_score, 5, 10)
            ap += average_precision(sorted_batch, pos_score)
            rr += reciprocal_rank(sorted_batch, pos_score)
            length += 1
    assert length != 0

    return (p_at_1_in_2/length, p_at_1_in_10/length, p_at_2_in_10/length, p_at_5_in_10/length, ap/length, rr/length) # R_1@2, R_1@10, R_2@10, R_5@10, MAP, MRR

if __name__ == '__main__':
    a = [[[5, 3, 2, 6, 9, 4, 12, 2, 5, 7]]]
    print(eval_samples(a))
