"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np


def uniform_embedding(shape, scale=0.0001):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def w2v_lookup(alphabet, word2vec_model, use_binary=True, augment_alphabet=False):
    """
    Create a word2vec lookup table with the word2vec vectors.
    :param alphabet: The alphabet that stores the words.
    :param word2vec_model: The word2vec model path.
    :param use_binary: Whether the word2vec model is binary.
    :param augment_alphabet: Whether to augment the alphabet with pre-trained word vectors.
    :return: A numpy array of shape [vocabulary size, dimension], each row is a word embedding.
    """
    print("Loading word2vec ...")
    model = Word2Vec.load_word2vec_format(word2vec_model, binary=use_binary)

    if augment_alphabet:
        for w in model.index2word:
            alphabet.add(w)

    table = np.empty([alphabet.size(), model.vector_size])

    table[alphabet.default_index, :] = uniform_embedding([1, model.vector_size])

    for w, index in alphabet.iteritems():
        if w in model:
            embedding = model[w]
        else:
            embedding = uniform_embedding([1, model.vector_size])
        table[index, :] = embedding

    print("Loading done...")
    return table
