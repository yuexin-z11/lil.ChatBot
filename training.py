import numpy as np
import random
import json
import jsonlines
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from utilities import bag_of_words, tokenize, stem
from model import NeuralN

# load data
intents = []
with jsonlines.open('./source/intent.jsonl', 'r') as reader:
    for obj in reader:
        intents.append(obj)

# initialize lists for storing words
all_wordss = []
tags = []
xy = []

# process intents
for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_wordss.extend(words)
        xy.append((words, tag))

# stem each word, remove duplicates, and sort 
ignore_words = ['?', '.', '!']
all_wordss = [stem(w) for w in all_wordss if w not in ignore_words]
all_wordss = sorted(set(all_wordss))
tags = sorted(set(tags))

# print(all_wordss)
# print(tags)
# print(xy)

# prepare taining data
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_wordss)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# actual training 
# variables
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

# define dataset class 
class ChatDataset(Dataset):
    def __init__(self):
        self.n_sample = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model, loss function, and optimizer
model = NeuralN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss() # loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        output = model(words)
        loss = criterion(output, labels)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# save model data
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
