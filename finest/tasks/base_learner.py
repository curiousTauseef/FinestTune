"""
A base learner that does not have any architecture. Implementations need to fill that part in.
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import os
from finest.utils.callbacks import MonitorLoss
import finest.utils.utils as utils
from keras.utils.visualize_util import plot

from finest.utils.configs import ExperimentConfig

logger = utils.get_logger(__name__)


class BaseLearner(object):
    def __init__(self):
        self.logger = utils.get_logger('Learner')
        self._embedding_layer_name = 'embedding'

        self.__monitor = 'val_acc'
        self.__weights_name = 'weights.h5'
        self.__architecture_name = 'architecture.json'
        self.__model_graph_name = 'structure.png'

        self.model = self.setup()

    def setup(self):
        pass

    def train_with_validation(self, train_x, train_y, dev_data, early_stop=True):
        self.logger.info("Training the model with provided validation.")

        if early_stop:
            hist = self.model.fit(train_x, train_y, validation_data=dev_data, callbacks=[self.__get_early_stop()])
        else:
            hist = self.model.fit(train_x, train_y, validation_data=dev_data, callbacks=[self.__get_loss_monitor()])
        return hist

    def train(self, train_x, train_y, early_stop=True):
        """
        Train the data.
        :param train_x: A dict of named training data. Each value is a 2-D array of shape [nb_samples, window size],
        which represent all the windowed sequence.
        :param train_y: A dict of named training data labels, Each value is a 2-D array of shape [nb_samples,
        nb_classes], which are one hot vector of the output.
        :param early_stop: Whether to early stop the model.
        :return:
        """
        self.logger.info("Training the model.")
        if early_stop:
            hist = self.model.fit(train_x, train_y, validation_split=0.1, callbacks=[self.__get_early_stop()])
        else:
            hist = self.model.fit(train_x, train_y, callbacks=[self.__get_loss_monitor()])
        return hist

    def __get_early_stop(self):
        return EarlyStopping(monitor=self.__monitor, patience=2, verbose=1)

    def __get_loss_monitor(self):
        return MonitorLoss(logger=self.logger, monitor=self.__monitor)

    def test(self, test_x, test_y):
        self.logger.info("Testing the model.")
        res = self.model.evaluate(test_x, test_y)
        return res

    def save(self, model_directory, overwrite=False):
        """
        Save both the model architecture and the weights to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        self.presave(model_directory)
        self.postsave(model_directory, overwrite)

    def presave(self, model_directory):
        """
        Save the model architecture to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        try:
            open(os.path.join(model_directory, self.__architecture_name), 'w').write(self.model.to_json(indent=2))
        except Exception as e:
            self.logger.warn("Model structure is not saved due to: %s" % repr(e))
        plot(self.model, to_file=os.path.join(model_directory, self.__model_graph_name))

    def postsave(self, model_directory, overwrite=False):
        """
        Save the weights to the given directory.
        :param model_directory: Directory to save model and weights.
        :param overwrite: Whether to overwrite the model.
        :return:
        """
        self.model.save_weights(os.path.join(model_directory, self.__weights_name), overwrite=overwrite)

    def load(self, model_directory):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param model_directory: Directory to save model and weights
        :return:
        """
        self.set_architecture(open(os.path.join(model_directory, self.__architecture_name)).read())
        self.load_weights_from_file(os.path.join(model_directory, self.__weights_name))

    def set_architecture(self, architecture_json):
        self.model = model_from_json(architecture_json)

    def load_weights_from_file(self, weights):
        self.model.load_weights(weights)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_embedding_layer(self):
        return self.model.get_layer(self._embedding_layer_name)

    def augment_embedding(self, additional_weights):
        embedding_layer = self.model.get_layer(self._embedding_layer_name)
        # The embedding layer weights is a singleton list of only one weights element.
        old_embedding_weights = self.model.get_layer(self._embedding_layer_name).get_weights()[0]
        new_weights = np.vstack([old_embedding_weights, additional_weights])

        additional_embedding_size = additional_weights.shape[0]
        new_embedding_size = embedding_layer.input_dim + additional_embedding_size

        logger.info("Changing the embedding dimension into %d x %d." % (new_weights.shape[0], new_weights.shape[1]))

        # Replace the Keras model with the augmented model.
        augmented_model = utils.change_single_node_layer_weights(self.model, self._embedding_layer_name,
                                                                 [new_weights], input_dim=new_embedding_size)

        return augmented_model
