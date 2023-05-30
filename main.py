import multiprocessing
import os
import subprocess

def start_client(client_id, num_rounds, num_epochs, lr, receive_port, send_port):
    # This function starts a client process.
    subprocess.run(["python", "client.py", str(client_id), str(num_rounds), str(num_epochs), str(lr), str(receive_port), str(send_port)])

if __name__ == "__main__":
    
    # Define the number of clients and training rounds
    num_clients = 10
    num_rounds = 10
    num_epochs = 2
    lr = 0.01
    receive_port = 12363
    send_port = 12364

    # Start the server
    server_process = subprocess.Popen(["python", "server.py", str(num_clients), str(num_rounds), str(receive_port), str(send_port)])

    # Start all the clients
    for client_id in range(num_clients):
        # Start a client in a new process
        process = multiprocessing.Process(target=start_client, args=(client_id, num_rounds, num_epochs, lr, receive_port, send_port))
        process.start()
        
    # Wait for all clients to finish
    for process in multiprocessing.active_children():
        process.join()

    # Terminate the server
    server_process.terminate()
