
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import time
import argparse
import torch
import numpy as np



def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Args for recommendation system reinforce_gan model")

    parser.add_argument('--data_folder',type =str,default ='./data/',help = 'dataset_folder')
    parser.add_argument('--dataset',type = str,default = 'yelp',help='çhoose from yelp,tb or rsc')
    parser.add_argument('--save_dir',type = str,default = './save_dir/',help='save folder')

    parser.add_argument('--resplit', type=eval, default=False)
    parser.add_argument('--num_thread', type=int, default=10, help='number of threadings')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_itrs', type=int, default=2000, help='num of iterations for q learning')
    # might change later to policy_grad method with attetion rather than lstm
    parser.add_argument('--rnn_hidden_dim', type=int, default=20, help='LSTM hidden sizes')
    parser.add_argument('--pw_dim', type=int, default=4, help='position weight dim')
    parser.add_argument('--pw_band_size', type=int, default=20, help='position weight banded size (i.e. length of history)')


    parser.add_argument('--dims', type=str, default='64-64')
    parser.add_argument('--user_model', type=str, default='PW', help='architecture choice: LSTM or PW')
    # dont think that the PW model could be used atm

    opts = parser.parse_args(args)

    return opts