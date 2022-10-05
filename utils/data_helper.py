import torch
import numpy as np
import os
from utils.log_helper import save_obj, load_obj

def sigmod(x):
    return 1. / (1. + torch.exp(-1. * x))

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def derivation_sigmoid(x):
    return 1 / (x * (1 - x) + 1e-8)

def load_data(args, name):
    train_file = os.path.join(args.data_dir, name)
    load_dir = args.data_dir

    if not os.path.exists(train_file + '.pkl'):
        print('error: there exist no file-{}'.format(train_file))
        # exit()
    return load_obj(load_dir, name)

def load_train(args):
    return load_data(args, 'train')

def load_test(args):
    return load_data(args, 'test')

def load_eval(args):
    return load_data(args, 'eval')

def save_data(args, data, name):
    path = os.path.join(args.data_dir, name + '.pkl')
    save_dir = args.data_dir
    if os.path.exists(path):
        print('there already exists file-{}, saving data will be ignored'.format(path))
        return
    else:
        save_obj(data, save_dir, name)

def save_train(args, data):
    return save_data(args, data, 'train')

def save_eval(args, data):
    return save_data(args, data, 'eval')

def save_test(args, data):
    return save_data(args, data, 'test')
