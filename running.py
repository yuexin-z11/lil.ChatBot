import random
import json
import jsonlines
import torch
import numpy as np
from model import NeuralN
from utilities import bag_of_words, tokenize

# use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# upload file 
intents = []
with jsonlines.open('./source/intent.jsonl', 'r') as reader:
    for obj in reader:
        intents.append(obj)

# load the model 
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# interaction with you 
bot_name = "CC"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.8:
        for intent in intents:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Can you word it in another way/")