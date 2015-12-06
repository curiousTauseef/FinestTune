"""
Implement the Multi-layer perceptron based POS tagger, following the Senna approach.
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.optimizers import Adadelta
import finest.utils.lookup as lookup

pos_dim = 20  # TODO read from data
window_size = 5
embedding_size = 300
voca_size = 300000  # TODO read from data


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
                self.model.add(Embedding(output_dim=HParam.vector_dim, input_dim=voca_size, weights=embeddings)))
        # TODO check concat axis index.
        return Merge(layers, mode='concat', concat_axis=1)

    def train_with_validation(self, x_train, y_train):
        self.model.fit(x_train, y_train, validation_split=0.1)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def test(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)


class DataReader:
    def __init__(self):
        pass

    def read_conll(self, path):
        data = []
        words = []
        poses = []
        voca = set()
        pos_voca = set()
        with open(path) as f:
            for l in f:
                if l.strip() == "":
                    data.append((words[:], poses[:]))
                    words = []
                    poses = []
                parts = l.split()
                word = parts[1]
                pos = parts[4]
                words.append(word)
                poses.append(pos)
                voca.add(word)
                pos_voca.add(pos)
        return data, list(voca), pos_voca


def main():
    reader = DataReader()
    wsj_train, voca, pos_voca = reader.read_conll("../data/brown_wsj_conll/eng.train.wsj.original")
    wsj_test, _, _ = reader.read_conll("../data/brown_wsj_conll/eng.test.wsj.original")

    w2v_table = lookup.w2v_lookup(voca, "../data/word2vec")  # TODO download it.

    mlp = PosMlp()
    mlp.setup(w2v_table)

    # TODO convert data with a sliding window.
    mlp.train()


if __name__ == '__main__':
    main()
