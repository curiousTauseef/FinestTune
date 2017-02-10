import numpy as np


class BaseTuner:
    def __init__(self, embedding_layer_weights, all_embeddings, seen_alphabet):
        self.num_seen = seen_alphabet.size()
        self.seen_embeddings_original = all_embeddings[:self.num_seen]
        self.unseen_embeddings = all_embeddings[self.num_seen:]
        self.seen_embeddings_tuned = embedding_layer_weights[:self.num_seen]

    def get_tuned_weights(self):
        tuned_unseen = self.tune(self.unseen_embeddings)
        return np.vstack((self.seen_embeddings_tuned, tuned_unseen))

    def tune(self, unseen_embeddings):
        pass
