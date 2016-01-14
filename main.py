#!/usr/bin/python

import os

import finest.utils.data_processor as processor
import finest.utils.lookup as lookup
from finest.tasks.labeling_mlp import LabelingMlp
import pickle
import time
import datetime
import finest.utils.utils as utils
import argparse

word2vec_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"
# word2vec_path = "../data/word2vec/vectors.bin"
window_size = 5

model_output_path = "../models/FinestTue/pos_all_vec"
log_history = "logs_all_vec/"

train_data = "../data/POS-penn/wsj/split1/wsj1.train.original"
dev_data = "../data/POS-penn/wsj/split1/wsj1.dev.original"
test_data = "../data/POS-penn/wsj/split1/wsj1.test.original"

logger = utils.get_logger("Main")


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def test(mlp, word_alphabet, pos_alphabet):
    word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(test_data)
    x_test = processor.slide_all_sentences(word_sentences_test, word_alphabet, window_size)
    y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)
    score = mlp.test(x_test, y_test)
    logger.info("Evaluation score is " + score)


def main():
    parser = argparse.ArgumentParser(description='Tuning with multi-layer perceptrons')
    parser.add_argument('--random_test_vector', help='start test randomly', action='store_true')

    args = parser.parse_args()

    logger.info("Loading conll data.")
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(train_data)

    word_sentences_dev, pos_sentences_dev, _, _ = processor.read_conll(dev_data)

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    if args.random_test_vector:
        word_alphabet.stop_grow()
        pos_alphabet.stop_grow()

    x_dev = processor.slide_all_sentences(word_sentences_dev, word_alphabet, window_size)
    y_dev = processor.get_all_one_hots(pos_sentences_dev, pos_alphabet)

    embeddings = lookup.w2v_lookup(word_alphabet, word2vec_path)

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    mlp = LabelingMlp(logger=logger, embeddings=embeddings, pos_dim=pos_alphabet.size(),
                      vocabulary_size=word_alphabet.size(), window_size=window_size)

    # history = mlp.train(x_train, y_train)
    history = mlp.train_with_validation(x_train, y_train, x_dev, y_dev)

    logger.info("Saving models at " + model_output_path)
    ensure_dir(model_output_path)
    mlp.save(model_output_path)

    logger.info("Saving alphabets at " + model_output_path)
    word_alphabet.save(model_output_path)
    pos_alphabet.save(model_output_path)

    logger.info("Saving history at " + log_history)
    ensure_dir(log_history)
    t = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    pickle.dump(history, open("history_at_" + t, 'w'))


if __name__ == '__main__':
    main()
