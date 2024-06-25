import tensorflow as tf
from tensorflow.keras import layers, models

class NNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NNetwork, self).__init__()
        # self.l1 = layers.Dense(hidden_size, input_shape=(input_size,), kernel_initializer='glorot_uniform')
        # self.l2 = layers.Dense(hidden_size, kernel_initializer='glorot_uniform')
        self.l1 = layers.Dense(hidden_size, input_shape=(input_size,), activation='relu', kernel_initializer='glorot_uniform')
        self.l2 = layers.Dense(hidden_size, activation='relu', kernel_initializer='glorot_uniform')
        self.l3 = layers.Dense(num_classes, kernel_initializer='glorot_uniform')
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(0.2)
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()

    def call(self, x, training=False):
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
