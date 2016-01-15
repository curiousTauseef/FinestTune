#!/usr/bin/python

import os

import finest.utils.data_processor as processor
import finest.utils.lookup as lookup
from finest.tasks.vanilla_labeling_mlp import VanillaLabelingMlp, BaseLearner
from finest.tasks.auto_embedding_mlp import AutoEmbeddingMlp
from finest.utils.alphabet import Alphabet
import time
import datetime
import finest.utils.utils as utils
import argparse

word2vec_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"
# word2vec_path = "../data/word2vec/vectors.bin"
model_output_base = "../models/FinestTune/pos"

train_data = "../data/POS-penn/wsj/split1/wsj1.train.original"
dev_data = "../data/POS-penn/wsj/split1/wsj1.dev.original"

logger = utils.get_logger("Main")


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def train_vanilla_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet, window_size, model_out):
    mlp = VanillaLabelingMlp(embeddings=embeddings, pos_dim=pos_alphabet.size(), vocabulary_size=word_alphabet.size(),
                             window_size=window_size)
    presave(mlp, model_out, word_alphabet, pos_alphabet)

    # history = mlp.train(x_train, y_train)
    history = mlp.train_with_validation(x_train, y_train, x_dev, y_dev)

    postsave(mlp, model_out, word_alphabet, pos_alphabet, history)
    return mlp


def train_auto_embedding_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet, window_size,
                             model_output):
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

    with open("history_at_" + t, 'w') as history_out:
        history_out.write(history.history)


def save(model, model_output, word_alphabet, pos_alphabet, history):
    presave(model, model_output, word_alphabet, pos_alphabet)
    postsave(model, model_output, word_alphabet, pos_alphabet, history)


def read_models(data_name, model, use_random_test):
    logger.info("Loading models from disk.")

    models = {}

    models_to_load = ['auto', 'vanilla'] if model == 'all' else [model]

    for t in models_to_load:
        model = BaseLearner()
        model_dir = get_model_directory(data_name, t, use_random_test)
        model.load(model_dir)

        pos_alphabet = Alphabet('pos')
        word_alphabet = Alphabet('word')

        pos_alphabet.load(model_dir)
        word_alphabet.load(model_dir)

        models[t] = (model, pos_alphabet, word_alphabet)

    logger.info("Loading done.")

    return models


def get_model_directory(data_set_name, model_name, use_random_test):
    suffix = "_rand" if use_random_test else ""
    return os.path.join(model_output_base, data_set_name, model_name + suffix)


def test(trained_models, use_random_test, test_conll, window_size):
    logger.info("Testing condition - [Use random vector] : %s ; [Test Data] : %s ." % (use_random_test, test_conll))
    for model_name, (model, pos_alphabet, word_alphabet) in trained_models.iteritems():
        word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(test_conll)
        x_test = processor.slide_all_sentences(word_sentences_test, word_alphabet, window_size)
        y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)

        evaluate_results = model.test(x_test, y_test)
        result_as_list = ", ".join("%.4f" % f for f in evaluate_results)
        logger.info("Direct test results are [%s] by model %s." % (result_as_list, model_name))


def train(model_to_train, use_random_test, train_conll, dev_conll, window_size, data_name):
    logger.info("Loading CoNLL data.")
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(train_conll)

    training_alphabet_output = os.path.join(model_output_base, data_name)
    ensure_dir(training_alphabet_output)
    word_alphabet.save(training_alphabet_output, 'training_words')

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    if use_random_test:
        logger.info("Dev set word vectors are not added to alphabet.")
        word_alphabet.stop_auto_grow()

    word_sentences_dev, pos_sentences_dev, _, _ = processor.read_conll(dev_conll)

    x_dev = processor.slide_all_sentences(word_sentences_dev, word_alphabet, window_size)
    y_dev = processor.get_all_one_hots(pos_sentences_dev, pos_alphabet)

    embeddings = lookup.w2v_lookup(word_alphabet, word2vec_path, not use_random_test)

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    models = {}
    if model_to_train == 'vanilla' or model_to_train == 'all':
        model_output = get_model_directory(data_name, 'vanilla', use_random_test)
        mlp_model = train_vanilla_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet, word_alphabet,
                                      window_size, model_output)
        models['vanilla'] = (mlp_model, pos_alphabet, word_alphabet)
    if model_to_train == 'auto' or model_to_train == 'all':
        model_output = get_model_directory(data_name, 'auto', use_random_test)
        mlp_model = train_auto_embedding_mlp(x_train, y_train, x_dev, y_dev, embeddings, pos_alphabet,
                                             word_alphabet, window_size, model_output)
        models['auto'] = (mlp_model, pos_alphabet, word_alphabet)
    return models


def main():
    parser = argparse.ArgumentParser(description='Tuning with multi-layer perceptrons')
    parser.add_argument('--random_test_vector', help='Start unseen test randomly, do not use pre-trained vectors',
                        action='store_true')
    parser.add_argument('--model', choices=['vanilla', 'auto', 'all'], help='The models to train/test.', required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--window_size', default=5)
    parser.add_argument('--data_name', default='wsj_split1')
    parser.add_argument('--test_data')  # test_data = "../data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    if args.test:
        if args.test_data is None:
            parser.error("--test_data is required when --test flag is on.")

    if not (args.test or args.train):
        parser.error("No action requested, add --test or --train.")

    models = {}
    if args.train:
        # If training now, then we use all the trained models
        models = train(args.model, args.random_test_vector, train_data, dev_data, args.window_size, args.data_name)

    if args.test:
        if len(models) == 0:
            models = read_models(args.data_name, args.model, args.random_test_vector)

        test(models, args.random_test_vector, args.test_data, args.window_size)


if __name__ == '__main__':
    main()
