# CS 5787 Deep Learning Assignment #1 - DL Basics

Authors: Mitchell Krieger

This repository contains materials to complete the [second assigment](./CS%205787%20-%20EX%202.pdf) for CS 5787 at Cornell Tech in the Fall 2024 semester. There are two parts to this assignment, the theoretical part and the practical part. 

## Theoretical

The first part is theoretical. All of its materials including a LaTex file and its PDF output can be found in the [report](./report/) directory.

## Practical

The second part of the assignment is practical. It is contained in the [assignment1_practical.ipynb](./assignment1_practical.ipynb) notebook. 

### Setup
All code, including data download, model definitions, training loops, hyperparameter tuning and evaluation should be runable from top to bottom of the notebook with reproduceable results by creating a virtual environment, installing the packages in requirements.txt, and logging into Weights & Biases.

```bash
source -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
wandb login
```

If you do not login to Weights & Biases via the command line the first cell of the notebook will attempt to login again. If you do not log in, you must set `mode` argument the `wandb.init()` method to `offline`. 

If cuda or mps is available to train models on GPUs, the notebook will attempt to use cuda, otherwise it will default to using cpu. If you wish to use mps, uncomment that section in `Check for GPU Access`.

### Data

For this assignment we are using the Penn Tree Bank dataset avaiable in the `data` folder already split into training, validation and testing sets. This dataset contains cleaned text with a new sample on each line. The `PTBText` pytorch Dataset will handle loading the data into tensors, via the `Vocab` class. If a vocabulary is not provided, the `PTBText` class will contruct one for you, otherwise you can pass any instance of `Vocab` and set `build_vocab` to false to provide another (this is done for the test and validation sets in this experiment). 

```python
train = PTBText('/content/ptb.train.txt', device=device)
val = PTBText('/content/ptb.valid.txt', vocab=train.vocab, build_vocab=False, device=device)
test = PTBText('/content/ptb.test.txt', vocab=train.vocab, build_vocab=False, device=device)

```


The `PTBText` class will handle the creation of minibatches called for in [”Recurrent Neural Network Regularization” by Zaremba et al. (2015)](https://arxiv.org/pdf/1409.2329). The default number of minibatches is set in the constructor via `batch_size` and defaults to 20. Minibatches can be accessed by indexing the `minibatches` attribute of `PTBText`. Indexing the `PTBText` instance directly returns inputs and labels at the index's timestep for sequentially traversal set by the `seq_len` attribute (defaults to 20). 

```python
train = PTBText('/content/ptb.train.txt', device=device)
minibatch1 = train.minibatches[1] ## returns entire minibatch
inputs, labels = train[j] ## returns inputs and labels from minibatches at timestep j

# iterate through the entire dataset:
for j in range(dataset.chunk_size // dataset.seq_len):
    inputs, labels = train[j] 

```


### Models & Training

A `ZarembaRNN` pytorch nn.module is created with similar architecture yet flexible architecture to the orginal paper. The recurrent layers can be chosen by either passing in `'lstm'` or `'gru'` into the module's constructor. The model's embedding dimensions, number of hidden units and number of recurrent layers can be set in the module constructor. The default values for these are 200, 200 and 2 respectively. Dropout can also be passed here both in recurrent cell dropout (`lstm_dropout`) and between layer dropout `dropout`. Weights will be initialized uniformly between [-0.1,0.1].

Training & evaluation functions are also provided for easy training.


### Loading & saving

Trained weights have been saved to the `models` directory as `.pt` files using the simple `torch.save()` method. Weights can be loaded for prediction and testing by the following:

```python
model = ZarembaRNN(**kwargs) #or any module from above
model.load_state_dict(torch.load(f'./models/{model_name}.pth', weights_only=True))
model.eval()
```

Models can be tested after loading using:
```
sample = 'the financial outlook has become strong for a company that has had a tough'
model = ZamrembaRNN('lstm', len(train.vocab)).to(device)
model.load_state_dict(torch.load('/content/lstm_noreg.pth', weights_only=True))
model.eval()
with torch.no_grad():
    tokens = train.vocab.encode(sample.split())
    inputs = torch.LongTensor(tokens).unsqueeze(0).to(device)
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device), 
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    outputs, hidden = model(inputs, hidden)

    prediction = torch.argmax(outputs, dim=-1)
    print(" ".join(train.vocab.decode(prediction.squeeze().tolist())))
```
