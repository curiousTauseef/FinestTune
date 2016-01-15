"""
A base learner that does not have any architecture. Implementations need to fill that part in.
"""

from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import json
import os
from finest.utils.callbacks import MonitorLoss
import finest.utils.utils as utils
from keras.utils.visualize_util import plot


class BaseLearner(object):
    def __init__(self):
        self.logger = utils.get_logger('Learner')
        self.model = self._get_model()
        self.setup()
        self.__monitor = 'val_acc'
        self.__weights_name = 'weights.h5'
        self.__architecture_name = 'architecture.json'
        self.__model_graph_name = 'structure.png'

    def _get_model(self):
        pass

    def setup(self):
        pass

    def train_with_validation(self, x_train, y_train, x_val, y_val, early_stop=True):
        self.logger.info("Training the model with provided validation.")
        if early_stop:
            hist = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[self.__get_early_stop()],
                                  show_accuracy=True)
        else:
            hist = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), show_accuracy=True,
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
        if early_stop:
            hist = self.model.fit(x_train, y_train, validation_split=0.1, callbacks=[self.__get_early_stop()],
                                  show_accuracy=True)
        else:
            hist = self.model.fit(x_train, y_train, callbacks=[self.__get_loss_monitor()], show_accuracy=True)
        return hist

    def __get_early_stop(self):
        return EarlyStopping(monitor=self.__monitor, patience=2, verbose=1)

    def __get_loss_monitor(self):
        return MonitorLoss(logger=self.logger, monitor=self.__monitor)

    def test(self, x_test, y_test):
        self.logger.info("Testing the model.")
        return self.model.evaluate(x_test, y_test, show_accuracy=True)

    def save(self, model_directory):
        """
        Save both the model architecture and the weights to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        self.presave(model_directory)
        self.postsave(model_directory)

    def presave(self, model_directory):
        """
        Save the model architecture to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        try:
            open(os.path.join(model_directory, self.__architecture_name), 'w').write(self.model.to_json())
        except Exception as e:
            self.logger.warn("Model structure is not saved due to: %s" % repr(e))
        plot(self.model, to_file=os.path.join(model_directory, self.__model_graph_name))

    def postsave(self, model_directory):
        """
        Save the weights to the given directory.
        :param model_directory: Directory to save model and weights.
        :return:
        """
        self.model.save_weights(os.path.join(model_directory, self.__weights_name))

    def load(self, model_directory):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param model_directory: Directory to save model and weights
        :return:
        """
        self.model = model_from_json(open(os.path.join(model_directory, self.__architecture_name)).read())
        self.model.load_weights(os.path.join(model_directory, self.__weights_name))
