import argparse
import multiprocessing
import os
import subprocess

def start_client(client_id, num_rounds, num_epochs, lr, receive_port, send_port):
    # This function starts a client process.
    subprocess.run(["python", "client.py", str(client_id), str(num_rounds), str(num_epochs), str(lr), str(receive_port), str(send_port)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10, help="The number of clients.")
    parser.add_argument("--num_rounds", type=int, default=10, help="The number of training rounds.")
    parser.add_argument("--num_epochs", type=int, default=5, help="The number of epochs for each training round.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD optimizer.")
    parser.add_argument("--receive_port", type=int, default=12371, help="The port to receive data from the server.")
    parser.add_argument("--send_port", type=int, default=12372, help="The port to send data to the server.")
    args = parser.parse_args()
    
    # Start the server
    server_process = subprocess.Popen(["python", "server.py", str(args.num_clients), str(args.num_rounds), str(args.num_epochs), str(args.receive_port), str(args.send_port)])

    # Start all the clients
    for client_id in range(args.num_clients):
        # Start a client in a new process
        process = multiprocessing.Process(target=start_client, args=(client_id, args.num_rounds, args.num_epochs, args.lr, args.receive_port, args.send_port))
        process.start()
        
    # Wait for all clients to finish
    for process in multiprocessing.active_children():
        process.join()

    # Terminate the server
    server_process.terminate()
