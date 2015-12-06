"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np


def w2v_lookup(vocabulary, word2vec_model, binary=True, dimension=300):
    """
    Create a word2vec lookup table with the word2vec vectors.
    :param vocabulary:
    :param word2vec_model:
    :param binary:
    :param dimension:
    :return:
    """
    model = Word2Vec.load_word2vec_format(word2vec_model, binary)
    table = np.empty([len(vocabulary), dimension])
    for index, w in enumerate(vocabulary):
        # TODO replace missing with random
        embedding = model[w]
        table[index, :] = embedding
    return table
