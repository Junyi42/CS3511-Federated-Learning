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
parser.add_argument("port", type=int, help="The port to listen on.")
args = parser.parse_args()
# 全局模型
global_model = MLP()
# 定义客户端数量
num_clients = args.num_clients
num_rounds = args.num_rounds
# 创建一个TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定socket到端口
server_address = ('localhost', args.port)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)
# 监听连接
sock.listen(num_clients)

# 设置最大等待时间（秒）
timeout = 60

def handle_client(connection):
    try:
        # 接收数据
        data = b''
        while True:
            # 设置接收超时时间
            connection.settimeout(timeout)
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

    except socket.timeout:
        print("No new connection in the last {} seconds. Exiting.".format(timeout))
    finally:
        # 清理连接
        connection.close()
for _ in range(num_rounds):

    # 记录已连接的客户端数量
    connected_clients = 0

    for _ in range(num_clients):
        # 等待连接
        print('waiting for a connection')
        connection, client_address = sock.accept()
        print('connection from', client_address)

        # 处理连接
        handle_client(connection)

        # 增加已连接的客户端数量
        connected_clients += 1

        # 当所有客户端都连接并传输完模型参数后，开始发送全局模型给所有客户端
        if connected_clients == num_clients:
            # 计算平均模型参数
            for param_global in global_model.parameters():
                param_global.data /= num_clients

            # load test data
            test_dataset = torchvision.datasets.MNIST(
                    './data', train=False, download=True, 
                    transform=transforms.ToTensor()
                )

            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 测试全局模型
            test(global_model, test_loader)

            # 发送全局模型给所有客户端
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
