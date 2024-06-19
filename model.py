# import all the necessary libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer # reduces words to their base or root form 
lemmatizer = WordNetLemmatizer()
import json
import pickle 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random 


# create lists and load file 
words = []
classes = []
docs = []
ignore_words = ['?', '!']
df = open('./source/his_intent.jsonl', encoding = 'utf-8').read() # read json file 
intents = json.loads(df)

# iterate through each intent in the json data
for i in intents['intents']:
    for p in intent['patterns']:
        # tokenize each pattern into words
        w = nltk.word_tokenize(p)
        words.append(w)

        # add the words and tag 
        docs.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatization and normalization on words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# sort list 
words = sorted(list(set(words))) 
classes = sorted(list(set(classes)))

# Print the number of documents, classes, and unique lemmatized words
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)


# serialize words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initialize training data
trianing = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# prepare training and output data
train_x = list(training[:, 0])
train_y = list(trianing[:, 1])

print("Training data created")

# create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), Activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, Activation='relu'))
model.add(Dropout(0.5))
model.add(dense(len(train_y[0]), Activation='softmax'))

# compile model
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, matrics=['accuracy'])

# fit model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# save model
model.save('chatbot_model.h5', hist)

print("Model Saved!")
print()