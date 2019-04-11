from collections import deque

import numpy as np
from nltk import FreqDist

from src import handle_pickles

VOCAB_SIZE = 4
pickle_handler = handle_pickles.PickleHandler()

class ShiftWordsPerCW():
    def __init__(self, X, cw=4):
        self.window = cw
        self.X = X

    def shift_left(self, cw):
        x_left = deque(self.X)
        all_shifted_lists = list()
        while cw:
            cw -= 1
            x_left.append(' ')
            x_left.popleft()
            x_left_indexed = get_indexed_words(list(x_left), mode='use_vocab')
            all_shifted_lists.append(x_left_indexed)
        return all_shifted_lists

    def shift_right(self, cw):
        x_right = deque(self.X)
        all_shifted_lists = list()
        while cw:
            cw -= 1
            x_right.appendleft(' ')
            x_right.pop()
            x_right_indexed = get_indexed_words(list(x_right), mode='use_vocab')
            all_shifted_lists.append(x_right_indexed)
        return all_shifted_lists

    def shift_input(self):
        X_left = self.shift_left(cw=self.window)
        X_right = self.shift_right(cw=self.window)
        return X_left, X_right

def filter_unicodes(vocab_list):
    unicode_list = ['\u200d', '\u200b']
    vocab_list = [each for each in vocab_list if each[0] not in unicode_list]
    return vocab_list


def get_indexed_words(X, mode='build_vocab'):
    X_char = [list(word) for word in X if len(word) > 0]
    if mode == 'build_vocab':
        dist = FreqDist(np.hstack(X_char))
        _x_vocab = dist.most_common(VOCAB_SIZE)
        x_vocab = filter_unicodes(_x_vocab)
        x_idx2char = [word[0] for word in x_vocab]
        x_idx2char.insert(0, '$') # starting token
        x_idx2char.append('U') # OOV chars
        pickle_handler.pickle_dumper(x_idx2char, 'index_to_char_mapping')
    elif mode == 'use_vocab':
        x_idx2char = pickle_handler.pickle_loader('index_to_char_mapping')
    x_char2idx = {letter: idx for idx, letter in enumerate(x_idx2char)}
    X = [[x_char2idx[char] if char in x_char2idx else x_char2idx['U'] for (i, char) in enumerate(word)]
                for (j, word) in enumerate(X_char)]
    return X


if __name__ == "__main__":
    X = ['Hello', "am", "I", "Hello"]
    y = ['Hyallo', 'yam', 'yi', 'Hyallo']
    X = [each[::-1] for each in X]
    get_indexed_words(X, mode='train')