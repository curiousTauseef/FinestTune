"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np
from finest.utils import utils

logger = utils.get_logger(__name__)


def uniform_embedding(shape, scale=0.0001):
    return np.random.uniform(low=-scale, high=scale, size=shape)


class Lookup:
    def __init__(self, word2vec_model, use_binary=True):
        """
        :param word2vec_model: The word2vec model path.
        :param use_binary: Whether the word2vec model is binary.
        :return:
        """
        logger.info("Loading word2vec ...")
        self.model = Word2Vec.load_word2vec_format(word2vec_model, binary=use_binary)
        print("Loading done...")

    def w2v_lookup(self, alphabet):
        """
        Create a word2vec lookup table with the word2vec vectors.
        :param alphabet: The alphabet that stores the words.
        :return: A numpy array of shape [vocabulary size, dimension], each row is a word embedding.
        """
        # if augment_alphabet:
        #     logger.info("Augment alphabet with pretrained word vectors.")
        #     for w in model.index2word:
        #         alphabet.add(w)

        logger.info("Take a sub embedding table of size: %d." % alphabet.size())
        table = np.empty([alphabet.size(), self.model.vector_size])
        table[alphabet.default_index, :] = uniform_embedding([1, self.model.vector_size])

        max_index = 0

        for w, index in alphabet.iteritems():
            embedding = self.model[w] if w in self.model else uniform_embedding([1, self.model.vector_size])
            table[index, :] = embedding

            if index > max_index:
                max_index = index

        return table
