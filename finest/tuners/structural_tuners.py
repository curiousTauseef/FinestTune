__author__ = 'Hector Zhengzhong Liu'

from finest.tuners.base_tuner import BaseTuner


class StructuralTuner(BaseTuner):
    def __init__(self, embedding_layer_weights, all_embeddings, seen_alphabet):
        BaseTuner.__init__(self, embedding_layer_weights, all_embeddings, seen_alphabet)
        num_seen = embedding_layer_weights.shape[0]
        training_portion = all_embeddings[:num_seen]
        self.__train(training_portion)

    def tune(self, unseen_embeddings):
        return self.__tune(unseen_embeddings)

    def __train(self, training_portion):
        pass

    def __tune(self, test_portion):
        pass


class KNNTuner(StructuralTuner):
    def __train(self, training_portion):
        pass

    def __tune(self, test_portion):
        pass


class FittingTuner(StructuralTuner):
    def __train(self, training_portion):
        pass

    def __tune(self, test_portion):
        pass
