import torch

def fn_backward_hook(module, grad_input, grad_output):
    print('grad_input:{}'.format(grad_input))

    print('grad_output:{}'.format(grad_output))