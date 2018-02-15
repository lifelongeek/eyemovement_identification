# eyemovement_identification

Pytorch version

[Quick run]

1. git clone https://github.com/gmkim90/eyemovement_identification.git
2. adjust path in data/train_shuffle.csv,  data/valid_shuffle.csv
3. python train.py

[Data]

Train : W1S1, W1S2, W2S1, W2S2, W3S1

Valid : W3S2


[Model]

{LSTM-RNN}*2- {FC}*2


[Tendency]

Loss does not decrease


[TODO]

1. measure class accuracy
2. modify architecture
3. make test set
4. make test.py
