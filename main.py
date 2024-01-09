import torch as t 
from torch.utils.data import dataset,DataLoader
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as T 
from torchnet import meter 
import tensorboard 
from torch.utils import tensorboard 
import os 

import models

class DefaultConfig:
    load_model_path = None
    to_model_path = './mdoels/save/model.pth'
    train_data_path = './data/train'
    test_data_path = './data/test'

    model = 'resnet50'
    lr = 0.001
    weight_decay = 1e-5
    lr_decay = 0.1

    max_epoch = 30
    max_plateau = 3

opt = DefaultConfig()

def train(**kwargs):
    pass 

def test(**kwargs):
    pass 

def val(model,val_dataloader):
    pass 

def write_csv(results,file_name):
    pass 

if __name__ == '__main__':
    import fire
    fire.Fire()