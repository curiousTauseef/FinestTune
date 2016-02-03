"""
A base learner that does not have any architecture. Implementations need to fill that part in.
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import Sequential, Graph
import os
from finest.utils.callbacks import MonitorLoss
import finest.utils.utils as utils
from keras.utils.visualize_util import plot


class BaseLearner(object):
    def __init__(self, graph_mode=True):
        self.logger = utils.get_logger('Learner')
        self._embedding_layer_name = 'embedding'
        self._main_input_layer_name = 'input'
        self._main_output_layer_name = "prediction_output"
        self._graph_mode = graph_mode

        self.model = self._get_model()

        self.__monitor = 'val_acc'
        self.__weights_name = 'weights.h5'
        self.__architecture_name = 'architecture.json'
        self.__model_graph_name = 'structure.png'

        self.setup()

    def _get_model(self):
        if self._graph_mode:
            return Graph()
        else:
            return Sequential()

    def setup(self):
        pass

    def train_with_validation(self, x_train, y_train, x_dev, y_dev, early_stop=True):
        self.logger.info("Training the model with provided validation.")
        train_data = {self._main_input_layer_name: x_train, self._main_output_layer_name: y_train}
        dev_data = {self._main_input_layer_name: x_dev, self._main_output_layer_name: y_dev}
        if early_stop:
            if self._graph_mode:
                hist = self.model.fit(train_data, validation_data=dev_data, callbacks=[self.__get_early_stop()])
            else:
                hist = self.model.fit(x_train, y_train, validation_data=(x_dev, y_dev),
                                      callbacks=[self.__get_early_stop()])
        else:
            if self._graph_mode:
                hist = self.model.fit(train_data, validation_data=dev_data, callbacks=[self.__get_loss_monitor()])
            else:
                hist = self.model.fit(x_train, y_train, validation_data=(x_dev, y_dev),
                                      callbacks=[self.__get_loss_monitor()])

        return hist

    def train(self, x_train, y_train, early_stop=True):
        """
        Train the data.
        :param x_train: A 2-D array of shape [nb_samples, window size], which represent all the windowed sequence.
        :param y_train: A 2-D array of shape [nb_samples, nb_classes], which are one hot vector of the output.
        :param early_stop: Whether to early stop the model.
        :return:
        """
        self.logger.info("Training the model.")
        train_data = {self._main_input_layer_name: x_train, self._main_output_layer_name: y_train}
        if early_stop:
            if self._graph_mode:
                hist = self.model.fit(train_data, validation_split=0.1, callbacks=[self.__get_early_stop()])
            else:
                hist = self.model.fit(x_train, y_train, validation_split=0.1, callbacks=[self.__get_early_stop()],
                                      show_accuracy=True)
        else:
            if self._graph_mode:
                hist = self.model.fit(train_data, callbacks=[self.__get_loss_monitor()])
            else:
                hist = self.model.fit(x_train, y_train, callbacks=[self.__get_early_stop()], show_accuracy=True)

        return hist

    def __get_early_stop(self):
        return EarlyStopping(monitor=self.__monitor, patience=2, verbose=1)

    def __get_loss_monitor(self):
        return MonitorLoss(logger=self.logger, monitor=self.__monitor)

    def test(self, x_test, y_test):
        self.logger.info("Testing the model.")

        if self._graph_mode:
            test_data = {self._main_input_layer_name: x_test, self._main_output_layer_name: y_test}
            res = self.model.evaluate(test_data)
        else:
            res = self.model.evaluate(x_test, y_test, show_accuracy=True)
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
        # TODO cannot load the model we save??? Why duplicate layer?
        self.model = model_from_json(open(os.path.join(model_directory, self.__architecture_name)).read())
        self.model.load_weights(os.path.join(model_directory, self.__weights_name))

    def augment_embedding(self, additional_weights):
        embedding_node = self.model.nodes[self._embedding_layer_name]
        if embedding_node is None:
            raise RuntimeError("Embedding layer [%s] is not set." % self._embedding_layer_name)
        old_weights = embedding_node.get_weights()
        # TODO check how to stack.
        new_weights = np.vstack([old_weights, additional_weights])
        embedding_node.set_weights(new_weights)
