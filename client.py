import socket
import torch
import argparse
from utils import MLP, train, test  # Replace with your model and functions
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import io
import time
import dill

def federated_train_and_send(client_id, num_rounds, num_epochs, lr, dataloader, server_ip, receive_port, send_port):
    # Initialize the model
    model = MLP()
    print("Client {} initialized the model.".format(client_id+1))
    for r in range(num_rounds+1):
        # Load the global model parameters
        global_params = None
        if r > 0:
            global_params = receive_global_params(server_ip, receive_port)
            model.load_state_dict(global_params)
            print("Client {} received global model parameters.".format(client_id+1))

            if r == num_rounds:
                break

        # Train the model for num_epochs
        train(model, dataloader, num_epochs, lr)
        print("Client {} finished training round {}.".format(client_id+1, r+1))
        # Send the model parameters to the server
        send_params_to_server(model, server_ip, send_port)

def send_params_to_server(model, server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying...")
            time.sleep(1)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s.sendall(buffer.getvalue())
    s.close()

def receive_global_params(server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying...")
            time.sleep(1)
    params_bytes = b''
    while True:
        packet = s.recv(4096)
        if not packet:
            break
        params_bytes += packet
    buffer = io.BytesIO(params_bytes)
    buffer.seek(0)
    params = torch.load(buffer)
    s.close()
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client_id", help="The ID of the client.")
    parser.add_argument("num_rounds", help="The number of training rounds.")
    parser.add_argument("num_epochs", help="The number of epochs for each training round.")
    parser.add_argument("lr", help="Learning rate for SGD optimizer.")
    parser.add_argument("receive_port", help="The port of the server for receiving global model.")
    parser.add_argument("send_port", help="The port of the server for sending local model.")
    args = parser.parse_args()

    # set seed for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    batch_size = 32
    transform = transforms.ToTensor()
    with open("./private_data/Client"+str(int(args.client_id)+1)+".pkl",'rb') as f:
        train_dataset=dill.load(f)
    # train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    server_ip = "localhost"  # Replace with your server IP

    federated_train_and_send(int(args.client_id), int(args.num_rounds), int(args.num_epochs), float(args.lr), dataloader, server_ip, int(args.receive_port), int(args.send_port))
