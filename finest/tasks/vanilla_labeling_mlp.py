"""
Implement the Multi-layer perceptron based sequence tagger, following the Senna approach. It can be used for tasks like
POS and NER.
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adadelta
from keras.regularizers import l2
from finest.tasks.base_learner import BaseLearner


class HParams:
    def __init__(self):
        pass

    num_hidden_units = 300
    num_middle_layers = 3

    # Relu is only available in development branch of Theano at the moment
    hidden_activation = "relu"
    regularizer = l2(0.001)

    output_layer_type = 'softmax'
    output_loss = 'categorical_crossentropy'
    optimizer = Adadelta()


class VanillaLabelingMlp(BaseLearner):
    def __init__(self, logger, embeddings, pos_dim, vocabulary_size, window_size, fix_embedding=False):
        self.window_size = window_size
        self.embeddings = embeddings
        self.fix_embedding = fix_embedding
        self.embedding_size = embeddings.shape[1]
        self.vocabulary_size = vocabulary_size
        self.pos_dim = pos_dim

        super(VanillaLabelingMlp, self).__init__()

        self.logger.info("Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_size, self.vocabulary_size))

    def _get_model(self):
        return Sequential()

    def setup(self):
        self.logger.info("Setting up layers.")

        # Add embedding layer.
        self.model.add(Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                 weights=[self.embeddings], input_length=self.window_size,
                                 trainable=not self.fix_embedding))
        self.logger.info("Embedded sequence output is %s" % str(self.model.output_shape))
        self.model.add(Flatten())
        self.logger.info("Flattened output is %s" % str(self.model.output_shape))

        # Adding deep layers.
        for _ in range(HParams.num_middle_layers):
            self.model.add(
                    Dense(output_dim=HParams.num_hidden_units, init='uniform', W_regularizer=HParams.regularizer,
                          activation=HParams.hidden_activation))

        # Add output layer.
        self.model.add(
                Dense(output_dim=self.pos_dim, init='uniform', W_regularizer=l2(),
                      activation=HParams.output_layer_type))
        self.model.compile(loss=HParams.output_loss, optimizer=HParams.optimizer)
        self.logger.info("Done setting layers.")
