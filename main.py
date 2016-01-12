#!/usr/bin/python

import os

import finest.utils.data_processor as processor
import finest.utils.lookup as lookup
from finest.tasks.labeling_mlp import LabelingMlp
import logging
import sys
import pickle
import time
import datetime

word2vec_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"
# word2vec_path = "../data/word2vec/vectors.bin"
window_size = 5

model_output_path = "../models/FinestTue/pos"
log_history = "logs/"


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def main():
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    logger.info("Loading conll data.")
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(
            "../data/brown_wsj_conll/eng.train.wsj.original")

    word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(
            "../data/brown_wsj_conll/eng.test.wsj.original")

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    x_test = processor.slide_all_sentences(word_sentences_test, word_alphabet, window_size)
    y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    w2v_table = lookup.w2v_lookup(word_alphabet, word2vec_path)

    mlp = LabelingMlp(logger=logger, embeddings=w2v_table, pos_dim=pos_alphabet.size(),
                      vocabulary_size=word_alphabet.size(),
                      window_size=window_size)

    history = mlp.train(x_train, y_train)

    mlp.test(x_test, y_test)

    ensure_dir(model_output_path)
    mlp.save(model_output_path)

    ensure_dir(log_history)
    t = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    pickle.dump(history, open("history_at_" + t, 'w'))


if __name__ == '__main__':
    main()
