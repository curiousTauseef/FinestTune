"""
Lookup tables.
"""
from gensim.models.word2vec import Word2Vec
import numpy as np
from finest.utils import utils

logger = utils.get_logger("Lookup")


def uniform_embedding(shape, scale=0.0001):
    return np.random.uniform(low=-scale, high=scale, size=shape)


# def augment_lookup(alphabet, table, word2vec_model, use_binary=True):
#     """
#     Augment the lookup table with additional alphabet items.
#     :param alphabet: The extended alphabet.
#     :param table: The table to be augmented.
#     :param word2vec_model: The word2vec model path.
#     :param use_binary: Whether the word2vec model is binary.
#     """
#     existing_embedding_size = table.shape()[0]
#
#     if alphabet.size() > existing_embedding_size:
#         logger.info("Alphabet has grown, will update the embedding table.")
#         logger.info("Loading word2vec ...")
#         model = Word2Vec.load_word2vec_format(word2vec_model, binary=use_binary)
#
#         for index, word in alphabet.enumerate_items(existing_embedding_size):
#             embedding = model[word] if word in model else uniform_embedding([1, model.vector_size])
#             table[index, :] = embedding


def w2v_lookup(alphabet, word2vec_model, use_binary=True):
    """
    Create a word2vec lookup table with the word2vec vectors.
    :param alphabet: The alphabet that stores the words.
    :param word2vec_model: The word2vec model path.
    :param use_binary: Whether the word2vec model is binary.
    :return: A numpy array of shape [vocabulary size, dimension], each row is a word embedding.
    """
    logger.info("Loading word2vec ...")
    model = Word2Vec.load_word2vec_format(word2vec_model, binary=use_binary)

    # if augment_alphabet:
    #     logger.info("Augment alphabet with pretrained word vectors.")
    #     for w in model.index2word:
    #         alphabet.add(w)

    table = np.empty([alphabet.size(), model.vector_size])

    table[alphabet.default_index, :] = uniform_embedding([1, model.vector_size])

    for w, index in alphabet.iteritems():
        embedding = model[w] if w in model else uniform_embedding([1, model.vector_size])
        table[index, :] = embedding

    print("Loading done...")
    return table
