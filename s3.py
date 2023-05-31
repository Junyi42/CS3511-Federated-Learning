import argparse
import multiprocessing
import os
import subprocess
import utils
import numpy as np

if __name__ == "__main__":
    print("Train on stage 3")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=5, help="The number of clients.")
    parser.add_argument("-r", type=int, default=5, help="The number of training rounds.")
    parser.add_argument("-e", type=int, default=5, help="The number of epochs for each training round.")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate for SGD optimizer.")
    parser.add_argument("--receive_port", type=int, default=13117, help="CLient receive data from the server.")
    parser.add_argument("--send_port", type=int, default=13112, help="Client send data to the server.")
    args = parser.parse_args()
    
    server_process = subprocess.Popen(["python", "server.py", str(args.receive_port), str(args.send_port),str(args.c), str(args.r), str(args.e)])
    
    clients = np.random.choice(20, args.c, replace=False) # Randomly select c clients
    for client_id in clients:
        process = multiprocessing.Process(target = utils.start_client, args=(args.receive_port, args.send_port, args.r, args.e,client_id, args.lr))
        process.start()
    for process in multiprocessing.active_children():
        process.join()
        
    server_process.terminate()
