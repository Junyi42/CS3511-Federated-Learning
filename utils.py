import socket
import dill
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import io
import numpy as np
import time
import subprocess

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
def receive_models(num_clients,receive_port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('localhost', receive_port)
    print('starting up on {} port {}'.format(*server_address))
    soc.bind(server_address)
    soc.listen(num_clients)
    local_params = []
    for i in range(num_clients):
        connection, client_address = soc.accept()
        print( client_address, 'connected')
        data = b''
        while 1:
            received_data = connection.recv(1024)
            if not received_data: 
                break
            data += received_data
            
        buffer = io.BytesIO(data)
        buffer.seek(0)
        params = torch.load(buffer)
        connection.close()
        local_params.append(params)
    
    soc.close()
    return local_params

def send_models(num_clients,send_port,global_params):
   
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_address = ('localhost', send_port)
    print('Server started on {} with port{}'.format(*server_address))
    soc.bind(server_address)
    soc.listen(num_clients)
    for i in range(num_clients):
        connection, client_address = soc.accept()
        print('Client', client_address,' connected')
        # Send global model to the client
        buffer = io.BytesIO()
        torch.save(global_params, buffer)
        buffer.seek(0)
        connection.sendall(buffer.getvalue())
        connection.close()
    soc.close()

def receive(server_ip, server_port): #Clients receive data from server
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while 1:
        try:
            #print(" server_ip:",server_ip," sever_port:",server_port)
            soc.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print("Connection refused.")
            time.sleep(1)
    params_bytes = b''
    while 1:
        received_data = soc.recv(1024)
        if not received_data:
            break
        params_bytes += received_data
    buffer = io.BytesIO(params_bytes)
    buffer.seek(0)
    params = torch.load(buffer)
    soc.close()
    return params

def send(data,server_ip, send_port): #Clients send data to server
    # Send local params to server
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while 1:
        try:
            soc.connect((server_ip, send_port))
            break
        except ConnectionRefusedError:
            print("Connection refused.")
            time.sleep(1)
    buf = io.BytesIO()
    torch.save(data, buf)
    buf.seek(0)
    soc.sendall(buf.getvalue())
    soc.close()
    
def train_local_model(train_loader,global_model_state_dict,epochs,learning_rate,num_classes):
    local_model = MLP()
    local_model.load_state_dict(global_model_state_dict)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            one_hot_labels = F.one_hot(labels, num_classes).float()
            #one_hot_labels = torch.FloatTensor(one_hot_labels)
            probs = local_model(features)
            loss = criterion(probs, one_hot_labels)
            loss.backward()
            # update model parameters
            optimizer.step()
            
    return local_model.state_dict()

def load_client_dataset(batch_size):
    #load data
    train_dataset_clients = []
    for i in range(20):
        with open("./Client"+str(i+1)+".pkl",'rb') as f:
            train_dataset_clients.append( dill.load(f))
            
    dataloader_client = []
    for dataset in train_dataset_clients:
        dataloader_client.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    return dataloader_client

def load_test_dataset(batch_size):
    # load test data
    test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True, 
            transform=transforms.ToTensor())

    dataloader_test = DataLoader(test_dataset, batch_size, shuffle=False)
    return dataloader_test

def agggregate_local_models(local_model_state_dicts):
    avg_model = MLP()
    for i in range(len(local_model_state_dicts)):
        client_model = MLP()
        client_model.load_state_dict(local_model_state_dicts[i])
        for param_avg, param_client in zip(avg_model.parameters(), client_model.parameters()):
            if i==0:
                param_avg.data = param_client.data
            else:
                param_avg.data = param_avg.data + param_client.data

    for param_avg in avg_model.parameters():
        param_avg.data /= len(local_model_state_dicts)
        
    return avg_model.state_dict()

def test_global_model(model, test_loader):
    model.eval()
    final_predictions = []
    final_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            preds = model(features)
            predicted_classes = torch.argmax(preds, dim=-1)
            predicted_classes = predicted_classes.tolist()
            labels = labels.tolist()
            final_predictions.extend(predicted_classes)
            final_labels.extend(labels)

    final_predictions = torch.tensor(final_predictions)
    final_labels = torch.tensor(final_labels)

    accuracy = (final_predictions == final_labels).float().mean().item()
    #print('Global Model Accuracy: {:.2f}%'.format(accuracy * 100))
    return accuracy

def start_client(receive_port, send_port, num_rounds, num_epochs,client_id, lr,):
    subprocess.run(["python", "client.py",str(receive_port), str(send_port),  str(num_rounds), str(num_epochs), str(client_id),str(lr)])