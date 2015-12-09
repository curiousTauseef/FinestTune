import finest.utils.lookup as lookup
import finest.tasks.data_processor as processor
from finest.tasks.mlp_pos import PosMlp

# word2vec_path = "../data/word2vec/GoogleNews-vectors-negative300.bin"
word2vec_path = "../data/word2vec/vectors.bin"
window_size = 5
embedding_size = 200


def main():
    print "Loading conll data."
    word_sentences_train, pos_sentences_train, word_alphabet, pos_alphabet = processor.read_conll(
        "../data/brown_wsj_conll/eng.train.wsj.original")

    word_sentences_test, pos_sentences_test, _, _ = processor.read_conll(
        "../data/brown_wsj_conll/eng.test.wsj.original")

    print "Sliding window on the data."
    x_train = processor.slide_all_sentences(word_sentences_train, word_alphabet, window_size)
    y_train = processor.get_all_one_hots(pos_sentences_train, pos_alphabet)

    x_test = processor.slide_all_sentences(word_sentences_test, word_alphabet, window_size)
    y_test = processor.get_all_one_hots(pos_sentences_test, pos_alphabet)

    print "Training data dimension is %s, here is a sample:" % (str(x_train.shape))
    print x_train[0]

    print "Label data dimension is %s, here is a sample:" % (str(y_train.shape))
    print y_train[0]

    w2v_table = lookup.w2v_lookup(word_alphabet, word2vec_path, embedding_size)

    mlp = PosMlp(pos_dim=pos_alphabet.size(), embedding_size=embedding_size, vocabulary_size=word_alphabet.size(),
                 window_size=window_size)
    mlp.setup(w2v_table)

    mlp.train(x_train, y_train)
    mlp.test(x_test, y_test)


if __name__ == '__main__':
    main()
