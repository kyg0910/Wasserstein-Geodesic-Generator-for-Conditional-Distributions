import os
#import sys
#import imp
#import logging
import argparse

from data import loader
from train import train
from evaluate import evaluate
#from utils import random_horizontal_flip, normalize_for_alexnet

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main(config):

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.generation_dir):
        os.makedirs(config.generation_dir)

    #load and pre-process dataset
    x, c = loader(config)
    
    if config.mode=='train':
        #train the model
        train(x,c,config)
        
    elif config.mode=='eval':
        #evaluate the model
        evaluate(x,c,config)
        
  
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    #mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    
    # Loss configuration.
    parser.add_argument('--lambda_recon', type=float, default=100.0, help='lambda_recon')
    parser.add_argument('--lambda_match_zc', type=float, default=1.0, help='lambda_match_zc')
    parser.add_argument('--lambda_translation', type=float, default=0.0, help='lambda_translation')
    parser.add_argument('--lambda_match_xcc', type=float, default=0.0, help='lambda_match_xcc')
    parser.add_argument('--lambda_cycle', type=float, default=0.0, help='lambda_cycle')
    parser.add_argument('--lambda_transport', type=float, default=0.0, help='lambda_transport')
    parser.add_argument('--lambda_gp', type=float, default=0.0, help='lambda_gp')
    parser.add_argument('--lambda_label_pred', type=float, default=0.0, help='lambda_label_pred')
    
    # save-load configuration
    parser.add_argument('--data_path', type=str, default='FaceDetectedExtendedYaleB_share', help='data_path')
    parser.add_argument('--model_dir', type=str, default='CGwithOTpair_gpu1_res64_20201226_8pm(share)', help='model_dir')
    parser.add_argument('--generation_dir', type=str, default='CGwithOTpair_gpu1_res64_20201226_8pm(share)', help='generation_dir')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--n_iter', type=int, default=10000, help='number of total iterations for training')
    parser.add_argument('--iter_critic', type=int, default=5, help='iter_critic')
    parser.add_argument('--print_period', type=int, default=100, help='print_period')
    parser.add_argument('--init_lr', type=float, default=2e-4, help='init_lr') 
    parser.add_argument('--lr_update_period', type=int, default=100, help='lr_update_period') 
    parser.add_argument('--lr_decay_start_iter', type=int, default=0, help='lr_decay_start_iter') 
    
    # Model configuration.
    parser.add_argument('--zdim', type=int, default=64, help='zdim')
    
    config = parser.parse_args()
    print(config)
    main(config)
