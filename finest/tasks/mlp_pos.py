"""
Implement the Multi-layer perceptron based sequence tagger, following the Senna approach. It can be used for tasks like
POS and NER.
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, TimeDistributedMerge
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.optimizers import Adadelta
from keras.models import model_from_json
import json
import os


class HParams:
    def __init__(self):
        pass

    num_hidden_units = 300

    # Relu is only available in development at the moment
    hidden_activation = "relu"


class LabelingMlp:
    def __init__(self, embeddings, pos_dim, vocabulary_size, window_size, fix_embedding=False):
        self.model = Sequential()
        self.window_size = window_size
        self.embedding_size = embeddings.shape[1]
        self.vocabulary_size = vocabulary_size
        self.pos_dim = pos_dim
        self.__weights_name__ = "weights.h5"
        self.__architecture_name__ = "architecture"

        self.setup(embeddings, fix_embedding)

        print "Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_size, self.vocabulary_size)

    def setup(self, embeddings, fix_embedding=False):
        print "Setting up layers."
        # self.model.add(self.__window_lookup_layer(embeddings, fix_embedding))
        self.model.add(Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                 weights=[embeddings], input_length=self.window_size, trainable=not fix_embedding))
        print "Embedded sequence output is %s" % str(self.model.output_shape)
        self.model.add(Flatten())
        print "Flattened output is %s" % str(self.model.output_shape)
        self.model.add(Dense(output_dim=HParams.num_hidden_units, init='uniform'))
        self.model.add(Activation(HParams.hidden_activation))
        self.model.add(Dense(output_dim=HParams.num_hidden_units, init='uniform'))
        self.model.add(Activation(HParams.hidden_activation))
        self.model.add(Dense(output_dim=self.pos_dim, init='uniform'))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
        print "Done setting layers."

    # def __window_lookup_layer(self, embeddings, fix_embedding):
    #     layers = []
    #     for i in range(0, self.window_size):
    #         next_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
    #                                  weights=[embeddings], input_length=self.window_size, trainable=not fix_embedding)
    #         layers.append(next_embedding)
    #
    #     # Concate the layers with the last axis, i.e. the embedding dimension.
    #     window_layer = Merge(layers, mode='concat')
    #     print window_layer.output_shape
    #     return window_layer

    def train_with_validation(self, x_train, y_train, validation_split=0.1):
        print "Training the model with validation."
        self.model.fit(x_train, y_train, validation_split=validation_split)

    def train(self, x_train, y_train, early_stop=True):
        """
        Train the data.
        :param x_train: A 2-D array of shape [nb_samples, window size], which represent all the windowed sequence.
        :param y_train: A 2-D array of shape [nb_samples, nb_classes], which are one hot vector of the output.
        :param early_stop: Whether to early stop the model.
        :return:
        """
        print "Training the model."
        if early_stop:
            self.model.fit(x_train, y_train)
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            self.model.fit(x_train, y_train, validation_split=0.1, callbacks=[early_stopping])

    def test(self, x_test, y_test):
        print "Testing the model."
        self.model.evaluate(x_test, y_test)

    def save(self, model_directory):
        """
        Save both the model architecture and the weights to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        json.dump(self.model.to_json(), open(os.path.join(model_directory, self.__architecture_name__), 'w'))
        self.model.save_weights(os.path.join(model_directory, self.__weights_name__))

    def load(self, model_directory):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param model_directory: Directory to save model and weights
        :return:
        """
        self.model = model_from_json(open(os.path.join(model_directory, self.__architecture_name__)).read())
        self.model.load_weights(os.path.join(model_directory, self.__weights_name__))
