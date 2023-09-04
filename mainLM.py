import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import LSTModel
from LSTModel import
import Questionnaires_DataLoader
import string
import random
import re
import requests
import unidecode #TODO- might need to change to "utf-8" decoder.
#from google.colab import files

url = "https://github.com/tau-nlp-course/NLP_HW2/raw/main/data/shakespeare.txt"


all_characters = string.printable
n_characters = len(all_characters)  # our vocabulary size (|V| from the handout)

dataset_as_string = unidecode.unidecode(requests.get(url).content.decode())
n_chars_in_dataset = len(dataset_as_string)
print(f'Total number of characters in our dataset: {n_chars_in_dataset}')

chunk_len = 400
n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100  # (D_h from the handout)
num_layers = 1
lr = 0.005

model = LSTModel(n_characters, hidden_size, n_characters, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def random_chunk():
    start_index = random.randint(0, n_chars_in_dataset - chunk_len)
    end_index = start_index + chunk_len + 1
    return dataset_as_string[start_index:end_index]

# Turn a string into list of longs
def chars_to_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

#print(chars_to_tensor('abcDEF'))
def random_training_set():
    chunk = random_chunk()
    input = chars_to_tensor(chunk[:-1])
    target = chars_to_tensor(chunk[1:])
    return input, target

def train(model, data_loader, input, target):
    hidden = model.init_hidden()
    model.zero_grad()
    loss = 0
    for batch_idx, (batch_input, batch_target_labels) in enumerate(data_loader):
        print(f"Batch {batch_idx}: Data shape: {batch_input.shape}, Labels shape: {batch_target_labels.shape}")
        for word in batch_input:
            for c in range(chunk_len):
                output, hidden = model(input[c], hidden)
                loss += criterion(output, target[c].view(-1))




    loss.backward()
    optimizer.step()

    return loss.item() / chunk_len


def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = chars_to_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = F.softmax(output / temperature, dim=-1)
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = chars_to_tensor(predicted_char)

    return predicted


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {math.floor(s)}s'


def calculate_perplexity(model, data, temperature=0.8):
    total_log_prob = 0
    total_characters = 0

    with torch.no_grad():
        hidden = model.init_hidden()
        for i in range(len(text) - 1):
            input_ = chars_to_tensor(data[i])
            target = chars_to_tensor(data[i + 1])
            output, hidden = model(input_, hidden)
            output_softmax = F.softmax(output / temperature, dim=1)
            predicted_probs = output_softmax[0][target]
            log_prob = torch.log(predicted_probs)
            total_log_prob += log_prob.item()
            total_characters += 1
    average_log_prob = total_log_prob / total_characters
    perplexity = math.exp(-average_log_prob)
    return perplexity


def main():

    start = time.time()

    all_losses = []
    loss_avg = 0

    for epoch in range(1, n_epochs + 1):
        loss = train(*random_training_set())
        loss_avg += loss

        if epoch % print_every == 0:
            print(
                f'[time elapsed: {time_since(start)}  ;  epochs: {epoch} ({epoch / n_epochs * 100}%)  ;  loss: {loss:.4}]')
            print(evaluate('Wh', 200), '\n')  # generate text starting with 'Wh'

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    uploaded1 = files.upload()
    uploaded_file = list(uploaded1.values())[0]
    text = unidecode.unidecode(uploaded_file.decode())

    perplexity = calculate_perplexity(model, text)
    print(f'Perplexity: {perplexity:.4f}')


main()
