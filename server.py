import socket
import torch
import io
from utils import MLP, test
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num_clients", type=int, help="The number of clients.")
parser.add_argument("num_rounds", type=int, help="The number of training rounds.")
parser.add_argument("send_port", type=int, help="The port which client send the models.")
parser.add_argument("receive_port", type=int, help="The port which client receive the models.")
args = parser.parse_args()
# 全局模型
global_model = MLP()
# 定义客户端数量
num_clients = args.num_clients
num_rounds = args.num_rounds

def handle_client(connection):
    try:
        # 接收数据
        data = b''
        while True:
            packet = connection.recv(4096)
            if not packet: 
                break
            data += packet

        # 从二进制数据中加载模型参数
        buffer = io.BytesIO(data)
        buffer.seek(0)
        params = torch.load(buffer)

        # 载入模型参数
        model = MLP()
        model.load_state_dict(params)

        # 聚合模型参数
        for param_global, param_client in zip(global_model.parameters(), model.parameters()):
            param_global.data += param_client.data
    finally:
        # 清理连接
        connection.close()

def receive_models():
    # 创建一个TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定socket到端口
    server_address = ('localhost', args.receive_port)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    # 监听连接
    sock.listen(num_clients)
    for _ in range(num_clients):
        # 等待连接
        print('waiting for a connection')
        connection, client_address = sock.accept()
        print('connection from', client_address)

        # 处理连接
        handle_client(connection)
    # 关闭socket
    sock.close()

def send_models():
    # 创建一个TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定socket到端口
    server_address = ('localhost', args.send_port)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    # 监听连接
    sock.listen(num_clients)
    for _ in range(num_clients):
        # 等待连接
        print('waiting for a connection for send global model')
        connection, client_address = sock.accept()
        print('connection from for global model', client_address)

        # 将全局模型参数发送给客户端
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)
        connection.sendall(buffer.getvalue())

        # 清理连接
        connection.close()
    # 关闭socket
    sock.close()

for _ in range(num_rounds):
    # 记录已连接的客户端数量
    connected_clients = 0

    # Receive models
    receive_models()

    # 计算平均模型参数
    for param_global in global_model.parameters():
        param_global.data /= num_clients

    # load test data
    test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True, 
            transform=transforms.ToTensor()
        )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # test the global model
    test(global_model, test_loader)

    # Send models
    send_models()
