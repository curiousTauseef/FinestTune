#!/usr/bin/python

import os

import finest.utils.data_processor as processor
from finest.tasks.vanilla_labeling_mlp import VanillaLabelingMlp, BaseLearner
from finest.tasks.auto_embedding_mlp import AutoEmbeddingMlp
from finest.utils.alphabet import Alphabet
from finest.utils.lookup import Lookup
import finest.utils.utils as utils
import argparse

import sys

logger = utils.get_logger(__name__)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def train_model(model, x_train, y_train, x_dev, y_dev, pos_alphabet, word_alphabet, model_output, overwrite):
    presave(model, model_output, word_alphabet, pos_alphabet)
    if x_dev is not None and y_dev is not None:
        history = model.train_with_validation(x_train, y_train, x_dev, y_dev)
    else:
        history = model.train(x_train, y_train)
    postsave(model, model_output, overwrite)
    return model


def presave(model, model_output, word_alphabet, pos_alphabet):
    logger.info("Saving model structures at " + model_output)
    ensure_dir(model_output)
    model.presave(model_output)

    logger.info("Saving alphabets at " + model_output)
    word_alphabet.save(model_output)
    pos_alphabet.save(model_output)


def postsave(model, model_output, overwrite):
    logger.info("Saving model weights at " + model_output)
    ensure_dir(model_output)
    model.postsave(model_output, overwrite=overwrite)


def save(model, model_output, word_alphabet, pos_alphabet, overwrite):
    presave(model, model_output, word_alphabet, pos_alphabet)
    postsave(model, model_output, overwrite=overwrite)


def read_models(model_base, data_name, model):
    logger.info("Loading models from disk.")

    models = {}

    models_to_load = ['auto', 'vanilla'] if model == 'all' else [model]

    for t in models_to_load:
        model = BaseLearner()
        model_dir = get_model_directory(model_base, data_name, t)
        model.load(model_dir)

        pos_alphabet = Alphabet('pos')
        word_alphabet = Alphabet('word')

        pos_alphabet.load(model_dir)
        word_alphabet.load(model_dir)

        models[t] = (model, pos_alphabet, word_alphabet)

    logger.info("Loading done.")

    return models


def get_model_directory(model_output_base, data_set_name, model_name):
    return os.path.join(model_output_base, data_set_name, model_name)


def test(trained_models, lookup, oov_embedding, test_conll, window_size):
    logger.info("Testing condition - [OOV Vector] : %s ; [Test Data] : %s ." % (oov_embedding, test_conll))
    for model_name, (model, pos_alphabet, train_alphabet) in trained_models.iteritems():
        alphabet_for_test = train_alphabet.get_copy()

        if oov_embedding == "pretrained":
            alphabet_for_test.restart_auto_grow()
        elif oov_embedding == "random":
            alphabet_for_test.stop_auto_grow()

        original_alphabet_size = alphabet_for_test.size()
        logger.info("Original alphabet size is %d" % original_alphabet_size)

        word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(test_conll)
        x_test = processor.slide_all_sentences(word_sentences_test, alphabet_for_test, window_size)
        y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)

        test_model = model
        if oov_embedding == "pretrained":
            logger.info("New alphabet size is %d" % alphabet_for_test.size())
            #  A new embedding using the extended word alphabet.
            new_embeddings = lookup.w2v_lookup(alphabet_for_test)
            additional_embeddings = new_embeddings[original_alphabet_size:]
            logger.info("New embedding size is %d" % len(additional_embeddings))
            test_model = model.augment_embedding(additional_embeddings)

        evaluate_result = test_model.test(x_test, y_test)
        try:
            result_str = ", ".join("%.4f" % f for f in evaluate_result)
        except TypeError:
            result_str = "%.4f" % evaluate_result
        logger.info("Direct test results are [%s] by model %s." % (result_str, model_name))


def train(model_to_train, model_base, lookup, oov_embedding, train_path, dev_path, window_size, data_name, overwrite):
    logger.info("Loading CoNLL data.")
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(train_path)

    training_alphabet_output = os.path.join(model_base, data_name)
    ensure_dir(training_alphabet_output)
    word_alphabet.save(training_alphabet_output, 'training_words')

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    if oov_embedding == 'random':
        logger.info("Dev set word vectors are not added to alphabet.")
        word_alphabet.stop_auto_grow()

    x_dev, y_dev = None, None
    if dev_path:
        word_sentences_dev, pos_sentences_dev, _, _ = processor.read_conll(dev_path)
        x_dev = processor.slide_all_sentences(word_sentences_dev, word_alphabet, window_size)
        y_dev = processor.get_all_one_hots(pos_sentences_dev, pos_alphabet)

    # Alphabet stop growing.
    word_alphabet.stop_auto_grow()

    # A embedding subset from the word alphabet.
    embeddings = lookup.w2v_lookup(word_alphabet)

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    models = {}
    if model_to_train == 'vanilla' or model_to_train == 'all':
        model_output = get_model_directory(model_base, data_name, 'vanilla')
        mlp = VanillaLabelingMlp(embeddings=embeddings, pos_dim=pos_alphabet.size(),
                                 vocabulary_size=word_alphabet.size(), window_size=window_size)
        train_model(mlp, x_train, y_train, x_dev, y_dev, pos_alphabet, word_alphabet, model_output, overwrite)
        models['vanilla'] = (mlp, pos_alphabet, word_alphabet)
    if model_to_train == 'auto' or model_to_train == 'all':
        model_output = get_model_directory(model_base, data_name, 'auto')
        mlp = AutoEmbeddingMlp(embeddings=embeddings, pos_dim=pos_alphabet.size(),
                               vocabulary_size=word_alphabet.size(), window_size=window_size)
        train_model(mlp, x_train, y_train, x_dev, y_dev, pos_alphabet, word_alphabet, model_output, overwrite)
        models['auto'] = (mlp, pos_alphabet, word_alphabet)
    return models


def main():
    parser = argparse.ArgumentParser(description='Tuning with multi-layer perceptrons')
    parser.add_argument('--oov', choices=['random', 'pretrained'], help='Embedding for oov word', required=True)
    parser.add_argument('--model', choices=['vanilla', 'auto', 'all'], help='The models to train/test.', required=True)
    parser.add_argument('--train')  # "../data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "../data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "../data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--window_size', default=5)
    parser.add_argument('--data_name', default='wsj_split1')
    parser.add_argument('--overwrite', action='store_true')
    # word2vec_path = "../data/word2vec/vectors.bin"
    parser.add_argument('--word2vec', default='../data/word2vec/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--model_base', default='../models/FinestTune/pos')

    args = parser.parse_args()

    if not (args.test or args.train):
        parser.error("No action requested, add --test or --train.")

    oov_embedding = args.oov

    if args.train or not args.oov == 'random':
        lookup = Lookup(args.word2vec)
    else:
        lookup = None

    models = {}
    if args.train is not None:
        # If training flag is activated, we will use these trained models directly.
        models = train(args.model, args.model_base, lookup, oov_embedding, args.train, args.dev, args.window_size,
                       args.data_name, args.overwrite)

    if args.test is not None:
        if len(models) == 0:
            # Trying to load existing model.
            models = read_models(args.model_base, args.data_name, args.model)
        test(models, lookup, oov_embedding, args.test, args.window_size)


if __name__ == '__main__':
    main()
