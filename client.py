import socket
import torch
import argparse
from utils import MLP  # Replace with your model and functions
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import io
import time
import dill
import utils

parser = argparse.ArgumentParser()
parser.add_argument("receive_port", type=int, help="Where clients receive models.")
parser.add_argument("send_port", type=int, help="Where clients send model.")
parser.add_argument("num_rounds", type=int, help="No.of rounds.")
parser.add_argument("num_epochs", type=int, help="No.of epochs.")
parser.add_argument("client_id", type=int, help="Client ID.")
parser.add_argument("lr",type=float,help="Learning rate.")
args = parser.parse_args()
server_ip = "localhost"

dataloader_client = utils.load_client_dataset(batch_size=32) 

print("Client {} stars to train now.".format(args.client_id+1))
for r in range(args.num_rounds+1):
    time.sleep(5)
    global_params = utils.receive(server_ip, args.receive_port)
    print("Client {} received global model parameters.".format(args.client_id+1)) 
    if r<args.num_rounds:
        local_params = utils.train_local_model(dataloader_client[args.client_id],global_params, args.num_epochs, float(args.lr),10)       
        print("Round {}: client {} finished.".format(r+1,args.client_id+1))
        
        utils.send(local_params,server_ip, args.send_port)
        print("Round {}: client {} sended local params to server.".format(r+1,args.client_id+1))
    

