from keras.regularizers import l2
from keras.optimizers import Adadelta


class ExperimentConfig:
    def __init__(self):
        pass

    models = ["vanilla", "auto"]
    tuning_method = ['linear', 'non-linear', 'knn', 'fitting']
    word_vector = "word2vec"
    word_vector_path = "../data/word2vec/vectors.bin"
    # word_vector_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"

    model_base = "../models/FinestTune/"
    embedding_initial = ["random", "pretrained"]
    window_size = 5

    datasets = [
        # ("wsj1", "../data/POS-penn/wsj/split1/wsj1.train.original", "../data/POS-penn/wsj/split1/wsj1.dev.original",
        #  "../data/POS-penn/wsj/split1/wsj1.test.original"),
        ("wsj_sample", "../data/POS-penn/wsj/split1_sample/wsj1.train", "../data/POS-penn/wsj/split1_sample/wsj1.dev",
         "../data/POS-penn/wsj/split1_sample/wsj1.test")
    ]

    main_input_name = "task_input"
    main_output_name = "task_output"


class MLPConfig:
    def __int__(self):
        pass

    num_hidden_units = 300
    num_middle_layers = 3
    hidden_activation = "relu"
    regularizer = l2(0.001)

    label_output_layer_type = 'softmax'
    label_output_loss = 'categorical_crossentropy'
    label_loss_weight = 1

    optimizer = Adadelta()

    # Whether to update the embedding.
    fix_embedding = [False, True]


class AutoConfig:
    def __init__(self):
        pass

    auto_output_layer_type = 'softmax'
    auto_output_loss = 'categorical_crossentropy'
    auto_loss_weight = 1

    auto_options = ["linear", "non-linear"]

    auto_input_name = "auto_input"
    auto_output_name = "auto_output"

    optimizer = Adadelta()


class NonLinearMapperConfig:
    def __init__(self):
        pass

    hidden_layer_type = 'tanh'
    num_hidden_units = 300

    optimizer = Adadelta()


class LinearMapperConfig:
    def __init__(self):
        pass

    num_hidden_units = 300
