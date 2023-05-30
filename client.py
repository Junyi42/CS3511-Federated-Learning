import socket
import torch
import argparse
from utils import MLP, train, test  # Replace with your model and functions
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import io
import time
def federated_train_and_send(client_id, num_rounds, num_epochs, lr, dataloader, server_ip, server_port):
    # Initialize the model
    model = MLP()
    print("Client {} initialized the model.".format(client_id+1))
    for r in range(num_rounds+1):
        # Load the global model parameters
        global_params = None
        if r > 0:
            # wait for 10 seconds to receive global model parameters, this is to prevent connet to the server before the server is ready
            time.sleep(10)
            global_params = receive_global_params(server_ip, server_port)
            model.load_state_dict(global_params)
            print("Client {} received global model parameters.".format(client_id+1))

            if r == num_rounds:
                break

        # Train the model for num_epochs
        train(model, dataloader, num_epochs, lr)
        print("Client {} finished training round {}.".format(client_id+1, r+1))
        # Send the model parameters to the server
        send_params_to_server(model, server_ip, server_port)

def send_params_to_server(model, server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_ip, server_port))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        s.sendall(buffer.getvalue())

def receive_global_params(server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_ip, server_port))
        params_bytes = b''
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            params_bytes += packet
        # 从二进制数据中加载模型参数
        buffer = io.BytesIO(params_bytes)
        buffer.seek(0)
        params = torch.load(buffer)
        return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client_id", help="The ID of the client.")
    parser.add_argument("num_rounds", help="The number of training rounds.")
    parser.add_argument("num_epochs", help="The number of epochs for each training round.")
    parser.add_argument("lr", help="Learning rate for SGD optimizer.")
    parser.add_argument("server_port", help="The IP address of the server.")
    args = parser.parse_args()

    # Load the dataset
    batch_size = 32
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    server_ip = "localhost"  # Replace with your server IP
    server_port = int(args.server_port)  # Replace with your server port

    federated_train_and_send(int(args.client_id), int(args.num_rounds), int(args.num_epochs), float(args.lr), dataloader, server_ip, server_port)