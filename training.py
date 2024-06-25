import numpy as np
import random
import json
import jsonlines
import tensorflow as tf 
from utilities import bag_of_words, tokenize, stem
from model import NNetwork

# load data
intents = []
with jsonlines.open('./source/his_intent.jsonl', 'r') as reader:
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
num_epochs = 2000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 128
output_size = len(tags)

# define dataset class 
class ChatDataset(tf.data.Dataset):
    def _generator():
        for features, label in zip(x_train, y_train):
            yield features, label

    def __new__(cls, *args, **kwargs):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(input_size,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )

dataset = ChatDataset()
tf_dataset = dataset.shuffle(len(x_train)).batch(batch_size)

# use gpu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# initialize model, loss function, and optimizer
model = NNetwork(input_size, hidden_size, output_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# for epoch in range(num_epochs):
#     for(words, labels) in tf_dataset:
#         with tf.GradientTape() as tape:
#             outputs = model(words, training = True)
#             loss = loss_fn(labels, outputs)

#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')

# save model data
data = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_wordss": all_wordss,
    "tags": tags
}
with open('metadata.json', 'w') as f:
    json.dump(data, f)

# Save model
model.save('model_tf') 
print(f'Training complete. Model saved.')
