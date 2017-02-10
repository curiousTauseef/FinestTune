"""
Implement the Multi-layer perceptron based sequence tagger, following the Senna approach. It can be used for tasks like
POS and NER.
"""

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.layers.embeddings import Embedding
from finest.tasks.base_learner import BaseLearner
import numpy as np
from finest.utils.configs import AutoConfig, MLPConfig, NonLinearMapperConfig, LinearMapperConfig, ExperimentConfig


class AutoEmbeddingMlp(BaseLearner):
    def __init__(self, embeddings, pos_dim, vocabulary_size, window_size, auto_option, fix_embedding=False):
        """
        :param auto_option:
        :param embeddings:
        :param pos_dim:
        :param vocabulary_size:
        :param window_size: The window is assume to be an odd number, so the center word is the current focus.
        :param fix_embedding:
        """
        self.window_size = window_size
        self.embeddings = embeddings
        self.fix_embedding = fix_embedding
        self.embedding_size = embeddings.shape[1]
        self.vocabulary_size = vocabulary_size
        self.pos_dim = pos_dim
        self.auto_option = auto_option
        self._graph_mode = True

        self.center_filter = self.__set_up_center_filer()

        super(AutoEmbeddingMlp, self).__init__()

        self.logger.info("Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_size, self.vocabulary_size))

    def __set_up_center_filer(self):
        output_dim = self.embedding_size

        half_window_length = self.window_size / 2

        # Create the weights array.
        left_zeros = np.zeros((self.embedding_size * half_window_length, self.embedding_size))
        right_zeros = np.zeros((self.embedding_size * half_window_length, self.embedding_size))
        center_diag = np.diag(np.ones(self.embedding_size))

        weights = np.hstack((left_zeros, center_diag, right_zeros))

        # Create the bias array.
        biases = np.zeros(output_dim)

        center_filter = [weights, biases]

        return center_filter

    def setup(self):
        self.logger.info("Setting up layers.")

        # Graph model to represent the auto embedding model.
        # We have two outputs: the pos output, and the auto encoder output for the center word.
        # We have window_size inputs: one for each word in window, they will go through the embedding layer, and we
        # concatenate them for the labelling network, and pick the middle one for the auto encoding network.

        # The input is a window of indices specifying the vocabulary.
        inputs = Input(shape=(self.window_size,), dtype='int32', name=ExperimentConfig.main_input_name)

        # Adding embedding layer, this will create a 3 dimension tensor for the whole sequence.
        embeddings = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                               weights=[self.embeddings], input_length=self.window_size,
                               trainable=not self.fix_embedding, name=self._embedding_layer_name)(inputs)

        # Flatten the embedding layer to get a window representation.
        flatten = Flatten()(embeddings)

        # Adding dense layers to create the multi-layer perceptron.
        prev_layer = flatten
        for i in range(MLPConfig.num_middle_layers):
            prev_layer = Dense(output_dim=MLPConfig.num_hidden_units, init='uniform',
                               W_regularizer=MLPConfig.regularizer,
                               activation=MLPConfig.hidden_activation)(prev_layer)

        # Creating the tagging output.
        label_output = Dense(output_dim=self.pos_dim, init='uniform', W_regularizer=MLPConfig.regularizer,
                             activation=MLPConfig.label_output_layer_type, name=ExperimentConfig.main_output_name
                             )(prev_layer)

        # Creating the auto-embedding line.
        # Apply masking on the embedding to get the central word embedding only, this will make embeddings other than
        # the center to be zero. A possible alternative is to use Lambda to slice it.
        flattened_embedding_size = self.embedding_size * self.window_size
        center_only = Dense(output_dim=flattened_embedding_size, input_dim=flattened_embedding_size,
                            weights=self.center_filter, trainable=False)(flatten)

        auto_dense = self.get_auto_layer()(center_only)

        auto_output = Dense(self.embedding_size, activation=AutoConfig.auto_output_layer_type,
                            name=ExperimentConfig.auto_output_name)(auto_dense)

        model = Model(input=inputs, output=[label_output, auto_output])

        model.compile(loss={ExperimentConfig.main_output_name: MLPConfig.label_output_loss,
                            ExperimentConfig.auto_output_name: AutoConfig.auto_output_loss},
                      loss_weights={ExperimentConfig.main_output_name: MLPConfig.label_loss_weight,
                                    ExperimentConfig.auto_output_name: AutoConfig.auto_loss_weight},
                      optimizer=AutoConfig.optimizer, metrics=['accuracy'])

        self.logger.info("Done setting layers for auto mlp.")

        return model

    def get_auto_layer(self):
        if self.auto_option == "linear":
            return Dense(output_dim=LinearMapperConfig.num_hidden_units)
        elif self.auto_option == "non-linear":
            return Dense(output_dim=NonLinearMapperConfig.num_hidden_units,
                         activation=NonLinearMapperConfig.hidden_layer_type)
