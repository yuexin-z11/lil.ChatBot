import tensorflow as tf
from tensorflow.keras import layers, models

class NNetwork(tf.keras.Model):
    def __init__(self, input, hidden, num_classes):
        # create the layers 
        super(NNetwork, self).__init__()
        self.l1 = layers.Dense(hidden, input_shape=(input, ))
        self.l2 = layers.Dense(hidden)
        self.l3 = layers.Dense(num_classes)
        self.relu = layers.ReLU() # activation function 
        self.dropout = layers.Dropout(0.5)
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()

    def call(self, x, training=False):
        # build the model
        x = self.l1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.l2(x)
        x = self.batch_norm2(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.l3(x)

        return x