import socket
import numpy as np
import sys
import utils
from utils import train_local_model
from utils import test_global_model
from utils import agggregate_local_models
from utils import MLP
import dill
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import multiprocessing
import os
import subprocess

def train_1_2(dataloader_client,dataloader_test,num_rounds,num_epochs,lr,mode='default',M=20):
    num_classes = 10
    global_model = MLP()
    global_model_path = './models/server_model.pth'
    torch.save(global_model.state_dict(), global_model_path)
    for r in (range(num_rounds)):
        if mode == 'default': # Stage 1: all clients participate in each round of updates
            idx = [i for i in range(len(dataloader_client))]
        else: # Stage 2: randomly select M clients
            idx = np.random.choice(len(dataloader_client), M, replace=False)
        
        # Train local models and save their state_dicts
        local_model_state_dicts = []
        for i in idx:
            print("Round:",r+1,"Client:",i+1)
            local_model_state_dict = train_local_model(dataloader_client[i], torch.load(global_model_path),num_epochs, lr, num_classes)
            save_path = "./models/client_models/model"+str(i+1)+'.pth' # e.g. ./models/client_models/model1.pth ~ model20.pth
            torch.save(local_model_state_dict, save_path)
            local_model_state_dicts.append(local_model_state_dict)
        
        # Aggregate local models and update global model
        global_model.load_state_dict(agggregate_local_models(local_model_state_dicts))
        torch.save(global_model.state_dict(), global_model_path)
        acc = test_global_model(global_model,dataloader_test)
        print("Round ",r+1," Acc = ",acc)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, default=5, help="Num of training rounds.")
    parser.add_argument("-e", type=int, default=5, help="Num of epochs.")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate of each client.")
    parser.add_argument("-M", type=int, default=20, help="The number of clients selected.")
    parser.add_argument("--b1",type=int,default=32,help="Batch size of clients.")
    parser.add_argument("--b2",type=int,default=100,help="Batch size of test data.")
    args = parser.parse_args()
    
    dataloader_client = utils.load_client_dataset(batch_size=args.b1)
    dataloader_test = utils.load_test_dataset(batch_size=args.b2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = MLP()
    global_model_path = './models/server_model.pth'
    global_model.load_state_dict(torch.load(global_model_path))

    print("train on stage 1")
    train_1_2(dataloader_client,dataloader_test,args.r,args.e,args.lr)
    acc1 = test_global_model(global_model,dataloader_test)

    print("train on stage 2")
    train_1_2(dataloader_client,dataloader_test,args.r,args.e,args.lr,mode='stage2',M=args.M) #none-default mode
    acc2 = test_global_model(global_model,dataloader_test)

    print("acc1:",acc1,"acc2:",acc2)