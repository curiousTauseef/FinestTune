"""
Implement the Multi-layer perceptron based sequence tagger, following the Senna approach. It can be used for tasks like
POS and NER.
"""

from keras.models import Sequential, Graph
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

    label_output_layer_type = 'softmax'
    layer_output_loss = 'categorical_crossentropy'

    auto_output_layer_type = 'softmax'
    auto_output_loss = 'mse'

    optimizer = Adadelta()


class LabelingMlp(BaseLearner):
    def __init__(self, logger, embeddings, pos_dim, vocabulary_size, window_size, fix_embedding=False):
        self.window_size = window_size
        self.embeddings = embeddings
        self.fix_embedding = fix_embedding
        self.embedding_size = embeddings.shape[1]
        self.vocabulary_size = vocabulary_size
        self.pos_dim = pos_dim

        super(LabelingMlp, self).__init__()

        self.logger.info("Pos Labels : %d, Embedding Dimension : %d, Vocabulary Size : %d" % (
            self.pos_dim, self.embedding_size, self.vocabulary_size))

    def _get_model(self):
        return Graph()

    def setup(self):
        self.logger.info("Setting up layers.")

        self.model = Graph()
        # Graph model with two outputs.

        concatenated_size = self.vocabulary_size * self.window_size

        self.model.add_input(name='input', input_shape=(self.vocabulary_size, self.window_size))

        # Add embedding layer.
        self.model.add_node(Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                      weights=[self.embeddings], input_length=self.window_size,
                                      trainable=not self.fix_embedding),
                            name='embedding', inputs='input'
                            )
        self.logger.info("Embedded sequence output is %s" % str(self.model.output_shape))
        self.model.add_node(Flatten(), name='flatten', inputs='embedding')
        self.logger.info("Flattened output is %s" % str(self.model.output_shape))

        # Adding deep layers.
        for i in range(HParams.num_middle_layers):
            input_layer_name = 'flatten' if i == 0 else "inner%d" % (i - 1)

            self.model.add_node(
                    Dense(output_dim=HParams.num_hidden_units, init='uniform', W_regularizer=HParams.regularizer,
                          activation=HParams.hidden_activation),
                    name='inner%d' % i, inputs=input_layer_name
            )

        # Add output layer for the tagging.
        self.model.add_node(
                Dense(output_dim=self.pos_dim, init='uniform', W_regularizer=HParams.regularizer,
                      activation=HParams.label_output_layer_type),
                name='pos_layer', inputs="inner%d" % (HParams.num_middle_layers - 1))
        self.model.add_output(name='pos_output', input='pos_layer')

        # Add output layer for the auto encoder.
        self.model.add_node(Dense(output_dim=concatenated_size, init='uniform',
                                  W_regularizer=HParams.regularizer, activation=HParams.auto_output_layer_type),
                            name='auto_layer', inputs='flatten')
        self.model.add_output(name='auto_output', input='auto_layer')

        self.model.compile(
                loss={'label_output': HParams.layer_output_loss, 'auto_output': HParams.auto_output_loss},
                optimizer=HParams.optimizer)

        self.logger.info("Done setting layers.")
