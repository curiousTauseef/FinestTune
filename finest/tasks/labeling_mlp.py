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
from finest.utils.callbacks import MonitorLoss


class HParams:
    def __init__(self):
        pass

    num_hidden_units = 300

    # Relu is only available in development at the moment
    hidden_activation = "relu"


class LabelingMlp:
    def __init__(self, logger, embeddings, pos_dim, vocabulary_size, window_size, fix_embedding=False):
        self.logger = logger

        self.model = Sequential()
        self.window_size = window_size
        self.embedding_size = embeddings.shape[1]
        self.vocabulary_size = vocabulary_size
        self.pos_dim = pos_dim
        self.__weights_name__ = "weights.h5"
        self.__architecture_name__ = "architecture"

        self.setup(embeddings, fix_embedding)

        self.logger.info("Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_size, self.vocabulary_size))

    def setup(self, embeddings, fix_embedding=False):
        self.logger.info("Setting up layers.")
        self.model.add(Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                 weights=[embeddings], input_length=self.window_size, trainable=not fix_embedding))
        self.logger.info("Embedded sequence output is %s" % str(self.model.output_shape))
        self.model.add(Flatten())
        self.logger.info("Flattened output is %s" % str(self.model.output_shape))
        self.model.add(Dense(output_dim=HParams.num_hidden_units, init='uniform'))
        self.model.add(Activation(HParams.hidden_activation))
        self.model.add(Dense(output_dim=HParams.num_hidden_units, init='uniform'))
        self.model.add(Activation(HParams.hidden_activation))
        self.model.add(Dense(output_dim=self.pos_dim, init='uniform'))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
        self.logger.info("Done setting layers.")

    def train_with_validation(self, x_train, y_train, validation_split=0.1):
        self.logger.info("Training the model with validation.")
        self.model.fit(x_train, y_train, validation_split=validation_split)

    def train(self, x_train, y_train, early_stop=True):
        """
        Train the data.
        :param x_train: A 2-D array of shape [nb_samples, window size], which represent all the windowed sequence.
        :param y_train: A 2-D array of shape [nb_samples, nb_classes], which are one hot vector of the output.
        :param early_stop: Whether to early stop the model.
        :return:
        """
        self.logger.info("Training the model.")
        if early_stop:
            monitor = 'val_loss'
            early_stopping = EarlyStopping(monitor=monitor, patience=2, verbose=1)
            monitor_loss = MonitorLoss(logger=self.logger, monitor=monitor)
            hist = self.model.fit(x_train, y_train, validation_split=0.1, callbacks=[early_stopping, monitor_loss])
        else:
            hist = self.model.fit(x_train, y_train)

        return hist

    def test(self, x_test, y_test):
        self.logger.info("Testing the model.")
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
