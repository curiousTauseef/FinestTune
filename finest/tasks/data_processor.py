import numpy as np
from collections import deque
from finest.utils.alphabet import Alphabet

padding_symbol = "##PADDING##"


def read_conll(path, padding_size=0):
    word_sentences = []
    pos_sentences = []
    words = []
    poses = []

    word_alphabet = Alphabet([padding_symbol])
    pos_alphabet = Alphabet([padding_symbol])

    paddings = [padding_symbol] * padding_size

    with open(path) as f:
        for l in f:
            if l.strip() == "":
                word_sentences.append(paddings + words[:] + paddings)
                pos_sentences.append(paddings + poses[:] + paddings)
                words = []
                poses = []
            else:
                parts = l.split()
                word = parts[1]
                pos = parts[4]
                words.append(word)
                poses.append(pos)
                word_alphabet.add(word)
                pos_alphabet.add(pos)

    return word_sentences, pos_sentences, word_alphabet, pos_alphabet


def sliding_window(instances, alphabet, window_size):
    num_slices = len(instances) - window_size + 1
    slided_data = np.empty([num_slices, 0])

    if window_size > len(instances):
        # This should not happen because of padding.
        raise IndexError("Window size cannot be larger than instances size.")

    window = deque()
    instance_index = 0
    while instance_index < window_size:
        window.append(instances[instance_index])
        instance_index += 1

    slice_index = 0
    while instance_index < len(instances):
        instance = window.popleft()
        for dim, instance_value in enumerate(instance):
            voca_index = alphabet.get_index(instance_value)
            slided_data[slice_index, 0] = voca_index
        window.append(instances[instance_index])
        instance_index += 1

    return slided_data
