"""
Implement the Multi-layer perceptron based sequence tagger, following the Senna approach. It can be used for tasks like
POS and NER.
"""

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from finest.tasks.base_learner import BaseLearner
from finest.utils.configs import MLPConfig, ExperimentConfig


class VanillaLabelingMlp(BaseLearner):
    def __init__(self, embeddings, pos_dim, vocabulary_size, window_size, fix_embedding=False):
        self.window_size = window_size
        self.embeddings = embeddings
        self.fix_embedding = fix_embedding
        self.embedding_dimension = embeddings.shape[1]
        self.pos_dim = pos_dim
        self.vocabulary_size = vocabulary_size

        super(VanillaLabelingMlp, self).__init__()

        self.logger.info("Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_dimension, self.vocabulary_size))

    def setup(self):
        self.logger.info("Setting up layers.")

        inputs = Input(shape=(self.window_size,), name=ExperimentConfig.main_input_name, dtype='int32')

        # Adding embedding layer, this will create a 3 dimension tensor for the whole sequence.
        embeddings = Embedding(output_dim=self.embedding_dimension, input_dim=self.vocabulary_size,
                               weights=[self.embeddings], input_length=self.window_size,
                               trainable=not self.fix_embedding, name=self._embedding_layer_name)(inputs)

        flatten = Flatten()(embeddings)

        # Adding deep layers.
        prev_layer = flatten
        for _ in range(MLPConfig.num_middle_layers):
            prev_layer = Dense(output_dim=MLPConfig.num_hidden_units, init='uniform',
                               W_regularizer=MLPConfig.regularizer, activation=MLPConfig.hidden_activation)(prev_layer)

        # Add output layer.
        label_output = Dense(output_dim=self.pos_dim, init='uniform', W_regularizer=l2(),
                             activation=MLPConfig.label_output_layer_type,
                             name=ExperimentConfig.main_output_name
                             )(prev_layer)

        model = Model(input=inputs, output=label_output)
        model.compile(loss=MLPConfig.label_output_loss, optimizer=MLPConfig.optimizer)

        self.logger.info("Done setting layers for vanilla MLP.")

        return model
