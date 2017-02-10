import numpy as np
from collections import deque
from finest.utils.alphabet import Alphabet
import finest.utils.utils as utils

padding_symbol = "##PADDING##"

logger = utils.get_logger(__name__)


def read_conll(path, label_alphabet=None):
    word_sentences = []
    pos_sentences = []
    words = []
    poses = []

    word_alphabet = Alphabet('word', (padding_symbol,))

    if label_alphabet is None:
        label_alphabet = Alphabet('label', (padding_symbol,))

    with open(path) as f:
        for l in f:
            if l.strip() == "":
                word_sentences.append(words[:])
                pos_sentences.append(poses[:])
                words = []
                poses = []
            else:
                parts = l.split()
                word = parts[1]
                pos = parts[4]
                words.append(word)
                poses.append(pos)
                word_alphabet.add(word)
                label_alphabet.add(pos)

    # Add the last sentence in.
    if len(words) > 0:
        word_sentences.append(words[:])
        pos_sentences.append(poses[:])

    return word_sentences, pos_sentences, word_alphabet, label_alphabet


def slide_sentence(words, alphabet, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd, otherwise there is no focus.")
    padding_size = window_size / 2
    paddings = [padding_symbol] * padding_size
    padded_words = paddings + words + paddings

    num_slices = len(words)
    slided_data = np.empty([num_slices, window_size], dtype=int)

    if window_size > len(padded_words):
        # This should not happen because of padding, unless there is no words.
        raise IndexError("Window size [%d] cannot be larger than instances size [%d], word size is [%d]." %
                         (window_size, len(padded_words), len(words)))

    # Create the window, initialized with [0: window_size]
    window = deque()
    window_right = 0
    while window_right < window_size:
        window.append(padded_words[window_right])
        window_right += 1

    slice_index = 0

    while window_right < len(padded_words):
        copy_window(window, alphabet, slided_data, slice_index)

        # Move the window to right.
        window.popleft()
        window.append(padded_words[window_right])
        window_right += 1
        slice_index += 1

    # Copy the missing last one.
    copy_window(window, alphabet, slided_data, slice_index)
    return slided_data


def copy_window(window, alphabet, slided_data, slice_index):
    # Copy content from the sliding window.
    for window_index, word in enumerate(window):
        voca_index = alphabet.get_index(word)
        slided_data[slice_index, window_index] = voca_index


def slide_all_sentences(sentences, alphabet, window_size):
    slice_list = []

    if len(sentences) == 0:
        raise IndexError("Number of sentences read is zero, cannot proceed.")

    for sentence in sentences:
        if len(sentence) > 0:
            slided_sentence = slide_sentence(sentence, alphabet, window_size)
            slice_list.append(slided_sentence)
    return np.vstack(slice_list)


def get_center_embedding(data, embedding_lookup):
    """
    Convert the window based instances to their center word's embedding.
    :param data:
    :param embedding_lookup:
    :return: Numpy array of the center word's embedding.
    """
    embedding_dimension = embedding_lookup.shape[1]
    data_size = data.shape[0]
    center_index = data.shape[1] / 2

    embeddings = np.zeros((data_size, embedding_dimension))

    for row, d in enumerate(data):
        embeddings[row, :] = embedding_lookup[d[center_index]]


def get_one_hot(instances, alphabet):
    """
    Transform the index based elements into an one-hot vector.
    :param instances: The list of the elements.
    :param alphabet: Lookup alphabet for the element's index.
    :return: Numpy array of one-hot vectors.
    """
    labels = np.zeros([len(instances), alphabet.size()])
    for index, instance in enumerate(instances):
        labels[index, alphabet.get_index(instance)] = 1
    return labels


def get_all_one_hots(instances_list, alphabet):
    """
    Turn the instance list to one hot vector matrix.
    :param instances_list:
    :param alphabet:
    :return:
    """
    all_labels = []
    for instances in instances_list:
        all_labels.append(get_one_hot(instances, alphabet))

    return np.vstack(all_labels)
