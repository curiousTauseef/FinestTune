__author__ = 'Hector Zhengzhong Liu'

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from finest.tuners.base_tuner import BaseTuner


class NNTuner(BaseTuner):
    def __init__(self, embedding_layer_weights, all_embeddings, seen_alphabet):
        BaseTuner.__init__(self, embedding_layer_weights, all_embeddings, seen_alphabet)
        embedding_dim = all_embeddings.shape[1]
        self.model = self.get_nn(embedding_dim)
        training_target = embedding_layer_weights[:seen_alphabet.size()]
        self.model.fit(self.seen_embeddings_original, training_target)

    def tune(self, unseen_embeddings):
        mapped_embeddings = self.model.predict(unseen_embeddings)
        return mapped_embeddings

    def get_nn(self, embedding_dim):
        pass


class NonLinearTuner(NNTuner):
    def get_nn(self, embedding_dim):
        num_hidden_units = 64
        inputs = Input(shape=(embedding_dim,))
        x = Dense(num_hidden_units, activation='tanh')(inputs)
        predictions = Dense(embedding_dim, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)
        model.compile(optimizer='Adadelta', loss='mse', metrics=['accuracy'])

        return model


class LinearTuner(NNTuner):
    def get_nn(self, embedding_dim):
        inputs = Input(shape=(embedding_dim,))
        predictions = Dense(embedding_dim, activation='softmax')(inputs)

        model = Model(input=inputs, output=predictions)
        model.compile(optimizer='Adadelta', loss='mse', metrics=['accuracy'])

        return model
