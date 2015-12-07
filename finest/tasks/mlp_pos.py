"""
Implement the Multi-layer perceptron based POS tagger, following the Senna approach.
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.optimizers import Adadelta
import finest.utils.lookup as lookup
import data_processor as processor
import os

pos_dim = 20  # TODO read from data
window_size = 5
embedding_size = 300
voca_size = 300000  # TODO read from data
word2vec_path = "/Users/zhengzhongliu/Documents/projects/data/word2vec/GoogleNews-vectors-negative300.bin"


class HParam:
    def __init__(self):
        pass

    num_hidden_units = 300
    vector_dim = 300
    activation = "relu"


class PosMlp:
    def __init__(self):
        self.model = Sequential()

    def setup(self, embeddings):
        self.model.add(self.__setup_window(embeddings))
        self.model.add(Dense(output_dim=HParam.vector_dim, init='uniform'))
        self.model.add(Activation(HParam.activation))
        self.model.add(Dense(output_dim=HParam.vector_dim, init='uniform'))
        self.model.add(Activation(HParam.activation))
        self.model.add(Dense(output_dim=pos_dim, init='uniform', activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adadelta())

    def __setup_window(self, embeddings):
        layers = []
        for i in range(0, window_size):
            layers.append(
                self.model.add(Embedding(output_dim=embedding_size, input_dim=voca_size, weights=embeddings)))
        # TODO check concat axis index.
        return Merge(layers, mode='concat', concat_axis=1)

    def train_with_validation(self, x_train, y_train):
        self.model.fit(x_train, y_train, validation_split=0.1)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def test(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)


def main():
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(
        "/Users/zhengzhongliu/Documents/projects/data/brown_wsj_conll/eng.train.wsj.original", window_size / 2)

    word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(
        "/Users/zhengzhongliu/Documents/projects/data/brown_wsj_conll/eng.test.wsj.original", window_size / 2)

    w2v_table = lookup.w2v_lookup(word_alphabet, word2vec_path)

    mlp = PosMlp()
    mlp.setup(w2v_table)

    x_train = processor.sliding_window(word_sentences_train, word_alphabet, window_size)
    y_train = processor.sliding_window(pos_sentences_train, pos_alphabet, window_size)

    x_test = processor.sliding_window(word_sentences_train, word_alphabet, window_size)
    y_test = processor.sliding_window(pos_sentences_test, pos_alphabet, window_size)

    mlp.train(x_train, y_train)
    mlp.test(x_test, y_test)


if __name__ == '__main__':
    main()
