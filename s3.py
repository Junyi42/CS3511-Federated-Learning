import argparse
import multiprocessing
import os
import subprocess
import utils
if __name__ == "__main__":
    print("Train on stage 3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=5, help="The number of clients.")
    parser.add_argument("--num_rounds", type=int, default=5, help="The number of training rounds.")
    parser.add_argument("--num_epochs", type=int, default=5, help="The number of epochs for each training round.")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate for SGD optimizer.")
    parser.add_argument("--receive_port", type=int, default=13117, help="CLient receive data from the server.")
    parser.add_argument("--send_port", type=int, default=13112, help="Client send data to the server.")
    args = parser.parse_args()

    server_process = subprocess.Popen(["python", "server.py", str(args.receive_port), str(args.send_port),str(args.num_clients), str(args.num_rounds), str(args.num_epochs)])
    for client_id in range(args.num_clients):
        process = multiprocessing.Process(target = utils.start_client, args=(args.receive_port, args.send_port, args.num_rounds, args.num_epochs,client_id, args.lr))
        process.start()
    for process in multiprocessing.active_children():
        process.join()
        
    server_process.terminate()
