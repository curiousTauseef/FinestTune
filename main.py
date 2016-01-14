#!/usr/bin/python

import os

import finest.utils.data_processor as processor
import finest.utils.lookup as lookup
from finest.tasks.vanilla_labeling_mlp import VanillaLabelingMlp
from finest.tasks.auto_embedding_mlp import AutoEmbeddingMlp
import pickle
import time
import datetime
import finest.utils.utils as utils
import argparse

word2vec_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"
# word2vec_path = "../data/word2vec/vectors.bin"
window_size = 5

model_output_base = "../models/FinestTune/pos"

data_name = "wsj_split1"

train_data = "../data/POS-penn/wsj/split1/wsj1.train.original"
dev_data = "../data/POS-penn/wsj/split1/wsj1.dev.original"
test_data = "../data/POS-penn/wsj/split1/wsj1.test.original"

logger = utils.get_logger("Main")


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def test(model, word_alphabet, pos_alphabet):
    word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(test_data)
    x_test = processor.slide_all_sentences(word_sentences_test, word_alphabet, window_size)
    y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)
    score = model.test(x_test, y_test)
    logger.info("Evaluation score is " + score)


def train_vanilla_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet, name_suffix=""):
    model_output = os.path.join(model_output_base, data_name, 'vanilla' + name_suffix)
    mlp = VanillaLabelingMlp(logger=logger, embeddings=embeddings, pos_dim=pos_alphabet.size(),
                             vocabulary_size=word_alphabet.size(), window_size=window_size)
    presave(mlp, model_output, word_alphabet, pos_alphabet)

    # history = mlp.train(x_train, y_train)
    history = mlp.train_with_validation(x_train, y_train, x_dev, y_dev)

    postsave(mlp, model_output, word_alphabet, pos_alphabet, history)
    return mlp


def train_auto_embedding_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet, name_suffix=""):
    model_output = os.path.join(model_output_base, data_name, 'auto_embedding' + name_suffix)
    mlp = AutoEmbeddingMlp(logger=logger, embeddings=embeddings, pos_dim=pos_alphabet.size(),
                           vocabulary_size=word_alphabet.size(), window_size=window_size)
    presave(mlp, model_output, word_alphabet, pos_alphabet)

    history = mlp.train_with_validation(x_train, y_train, x_dev, y_dev)
    save(mlp, model_output, word_alphabet, pos_alphabet, history)

    postsave(mlp, model_output, word_alphabet, pos_alphabet, history)

    return mlp


def presave(model, model_output, word_alphabet, pos_alphabet):
    logger.info("Saving model structures at " + model_output)
    ensure_dir(model_output)
    model.presave(model_output)

    logger.info("Saving alphabets at " + model_output)
    word_alphabet.save(model_output)
    pos_alphabet.save(model_output)


def postsave(model, model_output, word_alphabet, pos_alphabet, history):
    logger.info("Saving model weights at " + model_output)
    ensure_dir(model_output)
    model.postsave(model_output)

    logger.info("Saving history at " + model_output)
    ensure_dir(model_output)
    t = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    pickle.dump(history, open("history_at_" + t, 'w'))


def save(model, model_output, word_alphabet, pos_alphabet, history):
    presave(model, model_output, word_alphabet, pos_alphabet)
    postsave(model, model_output, word_alphabet, pos_alphabet, history)


def main():
    parser = argparse.ArgumentParser(description='Tuning with multi-layer perceptrons')
    parser.add_argument('--random_test_vector', help='Start unseen test randomly, do not use pre-trained vectors',
                        action='store_true')
    parser.add_argument('--model', choices=['vanilla', 'auto', 'all'], help='The models to run.', required=True)

    args = parser.parse_args()

    logger.info("Loading conll data.")
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(train_data)

    training_alphabet_output = os.path.join(model_output_base, data_name)
    ensure_dir(training_alphabet_output)
    word_alphabet.save(training_alphabet_output, 'training_words')

    word_sentences_dev, pos_sentences_dev, _, _ = processor.read_conll(dev_data)

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    if args.random_test_vector:
        logger.info("Dev/Test set word vectors are initialized randomly.")
        word_alphabet.stop_grow()
        pos_alphabet.stop_grow()

    x_dev = processor.slide_all_sentences(word_sentences_dev, word_alphabet, window_size)
    y_dev = processor.get_all_one_hots(pos_sentences_dev, pos_alphabet)

    embeddings = lookup.w2v_lookup(word_alphabet, word2vec_path)

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    if args.model == 'vanilla' or args.model == 'all':
        suffix = "rand" if args.random_test_vector else ""
        vanilla_mlp = train_vanilla_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet, suffix)
    if args.model == 'auto' or args.model == 'all':
        auto_mlp = train_auto_embedding_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet)


if __name__ == '__main__':
    main()
