import os
import torch


def select_device(device=''):
    device = str(device).strip().lower().replace('cuda', '').replace('none', '')
    cpu = device == 'cpu' or device == ""
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        os.environ[
            'CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()

    if not cpu and torch.cuda.is_available():
        device = device if device else '0'
        arg = 'cuda:0'
    else:
        arg = 'cpu'

    return torch.device(arg)
