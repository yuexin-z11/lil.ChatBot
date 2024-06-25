import random
import json
import jsonlines
import tensorflow as tf 
import numpy as np
from model import NNetwork
from utilities import bag_of_words, tokenize

# use gpu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# upload file 
intents = []
with jsonlines.open('./source/his_intent.jsonl', 'r') as reader:
    for obj in reader:
        intents.append(obj)

model = tf.keras.models.load_model('model_tf')

# extract model data
with open('metadata.json', 'r') as f:
    data = json.load(f)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']

# interaction with you 
bot_name = "CC"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    bow = np.array(x).reshape(1, -1)

    # prediction 
    output = model.predict(bow)
    predicted = np.argmax(output, axis=1)[0]
    tag = tags[predicted]

    # determine response based on confidence level
    confidence = tf.reduce_max(tf.nn.softmax(output)).numpy()
    count = 0
    if confidence > 0.1:
        for intent in intents:
            if tag == intent["tag"] and count != 1:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                count = 1
    else:
        print(f"{bot_name}: I do not understand...")