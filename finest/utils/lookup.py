"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np
from finest.utils import utils
from finest.utils.alphabet import Alphabet

logger = utils.get_logger(__name__)


def uniform_embedding(shape, scale=0.0001):
    return np.random.uniform(low=-scale, high=scale, size=shape)


class Lookup:
    def __init__(self, config):
        """
        :return:
        """
        if config.word_vector == 'word2vec':
            logger.info("Loading word2vec from disk ...")
            self.model = Word2Vec.load_word2vec_format(config.word_vector_path, binary=True)
        print("Loading done...")

        self.full_alphabet = Alphabet("full_lookup")

    def initail_lookup(self, alphabet):
        """
        Initialize the lookup table of the word vectors. This will create a full lookup table that contains all the
        vocabulary, and a table that contains only the given alphabet.
        :param alphabet: The alphabet that stores the words.
        :return: A numpy array of shape [vocabulary size, dimension], each row is a word embedding.
        """
        embeddings = []
        if Alphabet.default_index == 0:
            embeddings.append(uniform_embedding([1, self.model.vector_size]))
        else:
            raise ValueError("Default index is not the first one, you must change the implementation here.")

        # Add words from the given alphabet to the embedding list, and to the full alphabet.
        for w, index in alphabet.iteritems():
            if not self.full_alphabet.has_instance(w):
                embedding = self.model[w] if w in self.model else uniform_embedding([1, self.model.vector_size])
                embeddings.append(embedding)
                self.full_alphabet.add(w)

        # Store embeddings that appear in training data.
        self.table = np.vstack(embeddings)

        for w in self.model.vocab.keys():
            if not alphabet.has_instance(w):
                embedding = self.model[w]
                self.full_alphabet.add(w)
                embeddings.append(embedding)

        # Store embeddings of the full vocabulary.
        self.full_table = np.vstack(embeddings)

        logger.info("The training only embedding table contains %d embeddings, each with a dimension of size %d." % (
            self.table.shape[0], self.table.shape[1]))

        logger.info("The full embedding table contains %d embeddings, each with a dimension of size %d." % (
            self.full_table.shape[0], self.full_table.shape[1]))

    def load_additional_embeddings(self, original_alphabet, new_alphabet):
        """
        Create an additional lookup table that contains additional words that's not in the orginal ones.
        :param original_alphabet:  The original table.
        :param new_alphabet:  The additional table.
        :return:
        """
        embeddings = []
        for w, index in new_alphabet.iteritems():
            if not original_alphabet.has_instance(w):
                embedding = self.model[w] if w in self.model else uniform_embedding([1, self.model.vector_size])
                embeddings.append(embedding)

        if len(embeddings) > 0:
            additional_table = np.vstack(embeddings)
            return additional_table
        else:
            return None
