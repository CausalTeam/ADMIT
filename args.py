import os
import argparse
import numpy as np
from copy import copy
import time
import random

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser("dose-response estimation via neural network")
        parser.add_argument('--data', type=str, default='sim')
        parser.add_argument('--verbose', type=int, default=50, help='print train info freq')
        parser.add_argument('--n_data', type=int, default=5000)
        parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs to train')
        parser.add_argument('--batch_size', type=int, default=500)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--alpha', type=float, default=2)
        parser.add_argument('--init', type=float, default=0.01)
        parser.add_argument('--dynamic_type', type=str, default='power', choices=['power', 'mlp'])
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--log', action='store_true', default=False)
        parser.add_argument('--save_data', action='store_true', default=False)
        parser.add_argument('--load_data', action='store_true', default=False)
        parser.add_argument('--scale', action='store_true', default=False)
        
        self.args = parser.parse_args()
        if self.args.seed is None:
            self.args.seed = random.randint(0, 10000)

class Helper(Parser):
    def __init__(self):
        super(Helper, self).__init__()
        self.args.input_dim = 6
        n_train, n_eval = int(self.args.n_data * 0.67), int(self.args.n_data * 0.23)
        n_test = self.args.n_data - n_train - n_eval

        self.args.n_train = n_train
        self.args.n_test = n_test
        self.args.n_eval = n_eval

        local_time = time.strftime("%Y-%m-%d#%H%M%S",time.localtime(time.time()))
        self.args.local_time = local_time

        self.args.log_dir = './logs/{}/'.format(self.args.data)

        self.args.data_dir = './datasets/{}/'.format(self.args.data)
        self.args.model_dir = './models/{}/{}/'.format(self.args.data, self.args.local_time)

    @property
    def config(self):
        return self.args

    @property
    def args_to_dict(self):
        list_of_args = [
            "batch_size",            
            'data',            
            'n_train',
            'n_test',
            "learning_rate",
            "weight_decay",
            "dynamic_type",
            "seed",
            "dropout",           
            "load_data",
            "init",
            "scale",
        ]

        args_to_dict = dict(filter(lambda x: x[0] in list_of_args,
                                  self.args.__dict__.items()))
        
        return args_to_dict
