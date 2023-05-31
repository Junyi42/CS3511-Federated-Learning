# Fedarated Learing

## Stage 1 and Stage 2: Offline FL

Run python file `stage12.py` to train the model and get the result.

```bash
python stage12.py
```

## Stage 3: Online FL

Run python file `stage3_main.py` to train the model and get the result.

```bash
python stage3_main.py --num_clients 10 --num_rounds 10 --num_epochs 10 --lr 0.01 --receive_port 12377 --send_port 12378
```

Hyperparameters are

- `num_clients`: number of clients
- `num_rounds`: number of rounds
- `num_epochs`: number of epochs
- `lr`: learning rate
- `receive_port`: port for receiving params
- `send_port`: port for sending params

## Stage 4: Sweep

Just simply run `bash/sweep.sh` to train the model and get the result.

```bash
bash bash/sweep.sh
```