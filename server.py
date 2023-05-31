import socket
import torch
import io
from utils import MLP
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
import utils
import time

parser = argparse.ArgumentParser()
parser.add_argument("send_port", type=int, help="Where server sends model.")
parser.add_argument("receive_port", type=int, help="Where clients receive models.")
parser.add_argument("num_clients", type=int, help="No.of clients.")
parser.add_argument("num_rounds", type=int, help="No.of rounds.")
parser.add_argument("num_epochs", type=int, help="No.of epochs.")
args = parser.parse_args()

global_model = MLP()
utils.send_models(args.num_clients,args.send_port,global_model.state_dict())

for idx in range(args.num_rounds):
    # Receive models
    local_params = utils.receive_models(args.num_clients,args.receive_port)
    print("In round ",idx+1," received ",len(local_params)," clients")
    # Aggregate local models
    global_params = utils.agggregate_local_models(local_params)
    global_model.load_state_dict(global_params)
    # Send models
    utils.send_models(args.num_clients,args.send_port,global_params)
    # test the global model
    test_loader = utils.load_test_dataset(batch_size=100)
    acc=utils.test_global_model(global_model, test_loader)
    print("Round ",idx+1,", Acc = ",acc)
    # Record the result
    fn = './output/s3/' + str(args.num_rounds)+'_'+ str(args.num_epochs) 
    with open(fn, 'a') as file:
        file.write(f"Round {idx+1}, acc: {acc}\n")

    