import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model


class MLPAutoEncoder:
    """here is the legacy version of MLP based auto-encoder, used for legacy train,test, and hard_search"""

    def __init__(self):
        self.input_dim = 100
        self.dense1_dim = 75
        self.dense2_dim = 50
        self.dense3_dim = 75
        self.output_dim = 100
        self.activation_type = "relu"
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_signal = Input(shape=(self.input_dim,), dtype=tf.float32)
        dense1 = Dense(self.dense1_dim, activation=self.activation_type)(input_signal)
        dense2 = Dense(self.dense2_dim, activation=self.activation_type)(dense1)
        dense3 = Dense(self.dense3_dim, activation=self.activation_type)(dense2)
        output = Dense(self.output_dim, activation=None)(dense3)

        model = Model(inputs=input_signal, outputs=output)

        return model

