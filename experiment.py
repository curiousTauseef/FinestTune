#!/usr/bin/python

import argparse
import os

import finest.utils.data_processor as processor
import finest.utils.utils as utils
from finest.tasks.auto_embedding_mlp import AutoEmbeddingMlp
from finest.tasks.vanilla_labeling_mlp import VanillaLabelingMlp, BaseLearner
from finest.tuners.nn_tuners import NonLinearTuner, LinearTuner
from finest.tuners.structural_tuners import KNNTuner, FittingTuner
from finest.utils.alphabet import Alphabet
from finest.utils.configs import ExperimentConfig, AutoConfig, MLPConfig
from finest.utils.lookup import Lookup

import numpy as np

logger = utils.get_logger(__name__)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def train_model(model, train_x, train_y, dev_data, pos_alphabet, word_alphabet, model_output, overwrite):
    presave(model, model_output, word_alphabet, pos_alphabet)
    if dev_data is not None:
        history = model.train_with_validation(train_x, train_y, dev_data)
    else:
        history = model.train(train_x, train_y)
    postsave(model, model_output, overwrite)
    return model


def presave(model, model_output, word_alphabet, pos_alphabet):
    print model_output
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
        model_dir = os.path.join(model_base, data_name, t)
        model.load(model_dir)

        pos_alphabet = Alphabet('pos')
        word_alphabet = Alphabet('word')

        pos_alphabet.load(model_dir)
        word_alphabet.load(model_dir)

        models[t] = (model, pos_alphabet, word_alphabet)

    logger.info("Loading done.")

    return models


def enrich_embedding(model, all_embeddings):
    """
    Take the trained model and replace the embedding layer with the full vocabulary embedding.
    :param model:  The trained model.
    :param all_embeddings:  The full embedding model.
    :return:  The result model.
    """
    embedding_layer = model.get_embedding_layer()
    embedding_layer_weights = embedding_layer.get_weights()[0]

    combined_embeddings = np.vstack(all_embeddings[embedding_layer_weights.shape[0]:])

    return model


def fine_tune(model, fine_tune_method, all_embeddings, seen_alphabet):
    embedding_layer = model.get_embedding_layer()

    embedding_layer_weights = embedding_layer.get_weights()[0]

    logger.info("Fine tuning with %s method." % fine_tune_method)

    if fine_tune_method == 'linear':
        tuner = LinearTuner(embedding_layer_weights, all_embeddings, seen_alphabet)
    elif fine_tune_method == 'non-linear':
        tuner = NonLinearTuner(embedding_layer_weights, all_embeddings, seen_alphabet)
    elif fine_tune_method == 'knn':
        tuner = KNNTuner(embedding_layer_weights, all_embeddings, seen_alphabet)
    elif fine_tune_method == 'fitting':
        tuner = FittingTuner(embedding_layer_weights, all_embeddings, seen_alphabet)
    else:
        raise ValueError("Unknown fine tune method %s." % fine_tune_method)

    return model.augment_embedding(tuner.get_tuned_weights())


def test(trained_models, label_alphabet, lookup, oov_embedding, test_conll, window_size):
    logger.info("Testing condition - [OOV Vector] : %s ; [Test Data] : %s ." % (oov_embedding, test_conll))

    for model_name, (model, embedding_alphabet) in trained_models.iteritems():
        alphabet_for_test = embedding_alphabet.get_copy()
        original_alphabet_size = alphabet_for_test.size()
        logger.info("Original alphabet used to train the model is of size %d ." % original_alphabet_size)

        if oov_embedding == "pretrained":
            alphabet_for_test.restart_auto_grow()

        word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(test_conll)
        x_test = processor.slide_all_sentences(word_sentences_test, alphabet_for_test, window_size)
        y_test = processor.get_all_one_hots(pos_sentences_test, label_alphabet)

        logger.info("New alphabet size is %d" % alphabet_for_test.size())

        # TODO we seems need to make a copy of the model.
        test_model = model
        if oov_embedding == "pretrained":
            additional_embeddings = lookup.load_additional_embeddings(embedding_alphabet, alphabet_for_test)
            if additional_embeddings:
                logger.info("New embedding size is %d" % len(additional_embeddings))
                test_model = model.augment_embedding(additional_embeddings)

        evaluate_result = test_model.test(x_test, y_test)
        try:
            result_str = ", ".join("%.4f" % f for f in evaluate_result)
        except TypeError:
            result_str = "%.4f" % evaluate_result
        logger.info("Direct test results are [%s] by model %s." % (result_str, model_name))


def train(models_to_train, model_base, lookup, oov_handling, train_path, dev_path, window_size, data_name, overwrite):
    logger.info("Loading CoNLL data.")
    word_sentences_train, pos_sentences_train, word_alphabet, label_alphabet = processor.read_conll(train_path)

    # Take a snapshot of the current alphabet, which only contains training words. This is useful in fine tuning.
    train_alphabet = word_alphabet.get_copy()

    logger.info("Sliding window on the data.")
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, label_alphabet)

    label_alphabet.stop_auto_grow()

    if oov_handling == 'random':
        logger.info("Dev set word vectors are not added to alphabet.")
        word_alphabet.stop_auto_grow()
    else:
        # We will add development word embeddings to the alphabet so that their weights can be used.
        logger.info("Dev set word vectors will be added to alphabet.")

    x_dev, y_dev = None, None
    if dev_path:
        word_sentences_dev, pos_sentences_dev, _, _ = processor.read_conll(dev_path)
        x_dev = processor.slide_all_sentences(word_sentences_dev, word_alphabet, window_size)
        y_dev = processor.get_all_one_hots(pos_sentences_dev, label_alphabet)

    # Alphabet stop growing now anyways.
    word_alphabet.stop_auto_grow()

    logger.info("Training data dimension is %s, here is a sample:" % (str(x_train.shape)))
    logger.info(x_train[0])

    logger.info("Training label data dimension is %s, here is a sample:" % (str(y_train.shape)))
    logger.info(y_train[0])

    models = {}

    for model_name in models_to_train:
        lookup.initail_lookup(word_alphabet)

        if model_name == 'vanilla':
            model_output = os.path.join(model_base, data_name, 'vanilla')
            train_x = {ExperimentConfig.main_input_name: x_train}
            train_y = {ExperimentConfig.main_output_name: y_train}
            dev_data = ({ExperimentConfig.main_input_name: x_dev}, {ExperimentConfig.main_output_name: y_dev})

            for fix_embedding in MLPConfig.fix_embedding:
                mlp = VanillaLabelingMlp(embeddings=lookup.table, pos_dim=label_alphabet.size(),
                                         vocabulary_size=word_alphabet.size(), window_size=window_size,
                                         fix_embedding=fix_embedding)
                train_model(mlp, train_x, train_y, dev_data, label_alphabet, word_alphabet, model_output, overwrite)
                actual_model_name = model_name + "%s" % fix_embedding
                models[actual_model_name] = (mlp, word_alphabet)

        elif model_name == 'auto':
            if oov_handling == 'random':
                logger.info("We do not train the auto model when the embedding is initialized randomly.")
                continue

            train_x = {ExperimentConfig.main_input_name: x_train}

            y_auto_train = processor.get_center_embedding(x_train, lookup.table)
            y_auto_dev = processor.get_center_embedding(y_dev, lookup.table)

            train_y = {ExperimentConfig.main_output_name: y_train, AutoConfig.auto_output_name: y_auto_train}

            dev_data = ({ExperimentConfig.main_input_name: x_dev},
                        {ExperimentConfig.main_output_name: y_dev, AutoConfig.auto_output_name: y_auto_dev})

            for auto_option in AutoConfig.auto_options:
                model_output = os.path.join(model_base, data_name, 'auto', auto_option)
                mlp = AutoEmbeddingMlp(embeddings=lookup.full_table, pos_dim=label_alphabet.size(),
                                       vocabulary_size=lookup.full_alphabet.size(), window_size=window_size,
                                       auto_option=auto_option)
                train_model(mlp, train_x, train_y, dev_data, label_alphabet, lookup.full_alphabet, model_output,
                            overwrite)
                models[model_name + "_" + auto_option] = (mlp, lookup.full_alphabet)
        else:
            logger.warn("Unknown model name %s." % model_name)
            continue

    return models, train_alphabet, label_alphabet


def main():
    parser = argparse.ArgumentParser(description='Tuning with multi-layer perceptrons')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    for embedding_initial in ExperimentConfig.embedding_initial:
        logger.info("Running experiments with %s embedding initialization." % embedding_initial)

        lookup = None if ExperimentConfig.embedding_initial == 'random' else Lookup(ExperimentConfig)

        for data_name, train_data, dev_data, test_data in ExperimentConfig.datasets:
            logger.info("Running on data %s." % data_name)

            all_models = {}

            models_with_alphabet, train_alphabet, label_alphabet = train(ExperimentConfig.models,
                                                                         ExperimentConfig.model_base, lookup,
                                                                         embedding_initial, train_data, dev_data,
                                                                         ExperimentConfig.window_size, data_name,
                                                                         args.overwrite)
            for name, (model, alphabet) in models_with_alphabet.items():
                logger.info("Enriching embeddings for model.")
                all_models[name] = enrich_embedding(model, lookup.full_table)

                logger.info("Tuning model %s." % name)
                for fine_tune_method in ExperimentConfig.tuning_method:
                    all_models[name + "_" + fine_tune_method] = fine_tune(model, fine_tune_method, lookup.full_table,
                                                                          train_alphabet)

            # TODO see what is required for testing.
            test(models_with_alphabet, label_alphabet, lookup, embedding_initial, test_data,
                 ExperimentConfig.window_size)


if __name__ == '__main__':
    main()
