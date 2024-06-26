# ChatBot

## Project Overview
This ChatBot project was developed as a learning purpose focused on exploring machine learning techniques, specifically deep learning within the realm of natural language processing.

### Purpose
The primary goal of this project is educational. Due to limited training data, the ChatBot's current accuracy is modest, making it more of an educational tool than a robust question-answering system.

## Folder Structure and Files

### FirstVersion Folder
Initially, an attempt was made to create a ChatBot using Flask. However, this approach is not exactly what desired but it is an effort worth saving. 

### SecondTry Folder
This folder contains work from the second attempt. Here, everything was consolidated into a single file, and interactions were directly displayed on the terminal. `onefile.py` includes scripts for loading data, preprocessing, building the model, saving, and testing it. `model1` and `model1_improv` represent two different approaches to training the model in this file, though the results were not successful.

### ThirdVersion Folder
This folder contains all the files used in the third attempt. For this attempt, everything was organized by function to provide a clearer structure, still utilizing TensorFlow as the framework. The files build upon one another to create the ChatBot:
* `model.py` is solely for building the Neural Network,, clearly displaying each layer.
* `utilities.py` is for clearning up the words after loading the file including stem, bag of words. 
* `training.py` is a function to training the model and calling the two previous python script for the functions we need.
* `running.py` is for displayikng the interaction on terminal and running the model. 

### Chatbot Virtual Environment
The `Chatbot` directory serves as the virtual environment specifically created for running this project. This virtural environment is suitable for the first three versions of the ChatBot attempt.

### PypyChat Virtual Environment
The `pypyChat` directory is another virtual environment specificlly for the current attempt. This one contains packages me need including pytorch instead of tensorflow. 

### Current Version
All the files currently displayed here represent the third version of my work. This maintains a similar structure to the third attempt but switches to PyTorch from TensorFlow, as PyTorch is more suitable for projects like this. Thie final version finally achieved the goal for a almost 0 loss rate and has a good confidence after compiling. \\
To build and train the model:
``` python training.py ```
\\
To run and chat with the model:
``` python running.py ```

### Source Folder
Contains intent files used in the project. One before running the no_replicate file, the other is after. 

## Note
This is just a little fun thing for me to do and work on to familiarize myself with the skills I have enjoyed doing this and the model feedback is deisrable. It still needs work and I plan on expanding the dataset in the future. Overall, it is a really fun project to do and learn from. 

Lastly, 
Have fun, and thank you!