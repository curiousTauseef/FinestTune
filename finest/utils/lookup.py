"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np


def w2v_lookup(alphabet, word2vec_model, use_binary=True, dimension=300):
    """
    Create a word2vec lookup table with the word2vec vectors.
    :param alphabet: The alphabet that stores the words.
    :param word2vec_model: The word2vec model path.
    :param use_binary: Whether the word2vec model is binary.
    :param dimension: The dimension of the word2vec vectors.
    :return: A numpy array of shape [vocabulary size, dimension], each row is a word embedding.
    """
    print("Loading word2vec ...")
    model = Word2Vec.load_word2vec_format(word2vec_model, binary=use_binary)
    table = np.empty([alphabet.size(), dimension])
    for index, w in alphabet.iteritems():
        # TODO replace missing with random
        embedding = model[w]
        table[index, :] = embedding
    print("Loading done...")
    return table
