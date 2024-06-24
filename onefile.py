# following is to create a model
# based on jsonl model
# with a not good accuracy and a not so hot loss rate
# model still in training 
# after compiling this file go to the pre.py for the predictions
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow 
import random
from tensorflow import keras
import json

# set a variable
stemmer = LancasterStemmer()

# read the intents file 
data = []
with open('./source/his_intent.jsonl') as file:
    # parse jsonl file
    for line in file:
        data.append(json.loads(line))

# extract the data from the file 
# create variables
# this is preprocessing data
q = []
label = []
xd = []
yd = []
ignore_words = ['?', '!', '.', ',']

for intent in data:
    for pattern in intent['patterns']:
        ww = nltk.word_tokenize(pattern.lower()) # tokenize
        q.extend(ww)
        xd.append(ww)
        yd.append(intent['tag'])

    if intent['tag'] not in label:
        label.append(intent['tag'])

# word stemming
words = [stemmer.stem(w.lower()) for w in q if w not in ignore_words]
words = sorted(list(set(words)))
labels = sorted(label)

# bag of words 
# to see if the words exist in our volcaburary
train = []
out = []

for x, doc in enumerate(xd):
    bag = []

    wrd = [stemmer.stem(w.lower()) for w in doc]
    for w in q:
        if w in wrd:
            bag.append(1)
        else:
            bag.append(0)

    out_row = [0] * len(labels)
    out_row[labels.index(yd[x])] = 1

    train.append(bag)
    out.append(out_row)

# convert to numpy arrays
train = np.array(train)
out = np.array(out)

# print(train.shape)  # (733, 5194)
# print(out.shape)    # (733, 23)

# now the important part 
# developing a model
# use keras
model = keras.models.Sequential([
    keras.layers.Dense(1024, input_shape=(train.shape[1],), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(labels), activation='softmax')
])

# compile the model
# learning rate scheduling
scheduler = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=scheduler)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Save and fit the model
model.fit(train, out, epochs=100, batch_size=20, verbose=1)
model.save("model1_improv.h5")

# this function is to convert to a bag of words
def bag(sentence, words, max_words):
    bag = [0] * max_words  # Initialize bag with zeros
    
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for se in sentence_words:
        if se in words:
            bag[words.index(se)] = 1

    return np.array(bag)

# prediction function
def predict(sentence):
    input_bag = bag(sentence, words, train.shape[1])  
    input_bag = np.reshape(input_bag, (1, -1))
    
    prediction = model.predict(input_bag)
    predicted_index = np.argmax(prediction)
    
    if predicted_index < len(labels):
        tag = labels[predicted_index]

        for intent in data:
            if intent['tag'] == tag:
                responses = intent['responses']

        return random.choice(responses)

# time to interact 
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response = predict(user_input)
    print("Bot:", response)