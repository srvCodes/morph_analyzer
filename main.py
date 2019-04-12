import os
from nltk import FreqDist
import numpy as np
import re
import datetime
import sys
import gc
import pickle
import gzip

from collections import Counter, deque
from argparse import ArgumentParser
from src.hindi import extract_word_root_and_feature
import yaml
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from src import handle_pickles, process_words
from copy import copy, deepcopy
from keras.preprocessing.sequence import pad_sequences
from src.models import cnn_rnn_with_context
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

parser = ArgumentParser(description="Enter --lang = 'hindi' for Hindi and 'urdu' for Urdu; "
                                    "--mode = 'train, test, or predict'")
parser.add_argument("--lang", required=True)
parser.add_argument("--mode", required=True, default='test')
args = vars(parser.parse_args())
pickle_handler= handle_pickles.PickleHandler()

LANG, MODE, = args['lang'], args['mode']
CONFIG_PATH = 'config/'

VOCAB_SIZE = 89
CONTEXT_WINDOW = 4
FEATURE_NUMS = 8

def read_path_configs(filename):
    with open(CONFIG_PATH + filename, 'r') as stream:
        try:
            res = yaml.load(stream)
        except yaml.YAMLError as e:
            res = None
            print("Error while reading yaml: ", e)
    return res


class ProcessAndTokenizeData():
    def __init__(self, n_features, words, roots, features):
        self.n_features = n_features
        self.all_words, self.all_roots, self.all_segregated_features = words, roots, features


    def process_features(self):
        # list_of_counters = [Counter(each) for each in self.all_segregated_features]
        # labels = [list(each.keys()) for each in list_of_counters]
        if MODE == 'train':
            dict_of_encoders = {i: LabelEncoder() for i in range(self.n_features)}
            encoded_features = [value.fit_transform(self.all_segregated_features[key]) for key, value in
                                dict_of_encoders.items()]
            list_of_counters = [Counter(each) for each in encoded_features]
            class_labels = [each.keys() for each in list_of_counters]
            num_of_indiv_feature_tags = [max(each_cnt, key=int) + 1 for each_cnt in list_of_counters]
            categorical_features = [np_utils.to_categorical(feature, num_classes=n) for feature, n in
                                    zip(encoded_features, num_of_indiv_feature_tags)]
            _ = [pickle_handler.pickle_dumper(obj, name) for obj, name in zip([dict_of_encoders, num_of_indiv_feature_tags,
                                                                     categorical_features],
                                                      ["dict_of_encoders", "num_of_indiv_features",
                                                       "categorized_features"])]
            return categorical_features, num_of_indiv_feature_tags
        elif MODE == 'test':
            dict_of_encoders, num_of_indiv_feature_tags = [pickle_handler.pickle_loader(name) for name in
                                                           ["dict_of_encoders", "num_of_indiv_features"]]
            encoded_features_test = [dict_of_encoders[i].transform(self.all_segregated_features[i]) \
                                for i in range(self.n_features)]
            categorical_features_test = [np_utils.to_categorical(feature, num_classes=n) for feature, n in
                                         zip(encoded_features_test, num_of_indiv_feature_tags)]
            return categorical_features_test


    def process_words_and_roots(self, context_window=4):
        X = [item[::-1] for item in self.all_words]
        y = deepcopy(self.all_roots)
        X_indexed = process_words.get_indexed_words(X, mode='build_vocab', vocab_size=VOCAB_SIZE)
        y_indexed = process_words.get_indexed_words(y, mode='use_vocab', vocab_size=VOCAB_SIZE)
        X_indexed_left, X_indexed_right = process_words.ShiftWordsPerCW(X=X, cw=context_window)
        all_inputs = list()
        all_inputs.append(X_indexed)
        all_inputs += X_indexed_left
        all_inputs += X_indexed_right
        all_inputs.append(y_indexed)
        return all_inputs


def sequence_padder(in_list, maxlen):
    out_list = pad_sequences(in_list, maxlen=maxlen, dtype='int32', padding='post')
    return out_list


def pad_all_sequences(indexed_outputs):
    X_indexed, X_indexed_left, X_indexed_right, y_indexed = indexed_outputs
    max_word_len = max(max([len(word) for word in indexed_outputs[0]]), max([len(word) for word in indexed_outputs[-1]]))
    all_padded_inputs = [sequence_padder(each, max_word_len) for each in indexed_outputs]
    return all_padded_inputs, max_word_len

def _create_model(max_word_len, embed_dim, n):
    model_instance = cnn_rnn_with_context.MorphAnalyzerModels(max_word_len=max_word_len, vocab_len=VOCAB_SIZE,
                                                              embedding_dim=embed_dim, list_of_feature_nums=n,
                                                              cw=CONTEXT_WINDOW)
    compiled_model = model_instance.create_and_compile_model()
    return compiled_model

def split_train_val(all_data, train_size):
    train_data = [x[:train_size] for x in all_data]
    val_data = [x[train_size:] for x in all_data]
    return train_data, val_data

def get_decoder_input(x_train):
    x_decoder_input = np.zeros_like(x_train)
    x_decoder_input[:, 1:] = x_train[:, :-1]
    x_decoder_input[:, 0] = 1
    return x_decoder_input

# def one_hot_encode()
def segregate_inputs_and_outputs(words_and_roots, features, decoder_inputs):
    roots = words_and_roots[-1]
    inputs = words_and_roots[:-1]
    inputs.append(decoder_inputs)
    outputs = [roots]
    outputs.append(features)
    return inputs, outputs


def main():
    paths = read_path_configs('data_paths.yaml')
    if LANG == 'hindi':
        if MODE == 'train':
            train_data_dir = paths['hdtb']['train']
            train_words, train_roots, train_features = \
                extract_word_root_and_feature.get_words_roots_and_features(train_data_dir, n_features=FEATURE_NUMS)
            val_data_dir = paths['hdtb']['validation']
            val_words, val_roots, val_features = \
                extract_word_root_and_feature.get_words_roots_and_features(val_data_dir, n_features=FEATURE_NUMS)
            assert len(train_words) == len(train_roots) == len(train_features[1]), \
                "Length mismatch while flattening train features"
            assert len(val_words) == len(val_roots) == len(val_features[1]),\
                "Length mismatch while flattening val features"
            # print("words: {}, roots: {}, features: {}".format(words[:5], roots[:5], features[:5]))
            train_size, val_size = [len(each) for each in [train_words, val_words]]
            train_val_words, train_val_roots = [i + j for i,j in zip([train_words, train_roots], [val_words, val_roots])]
            train_val_features = [i+j for i,j in zip(train_features, val_features)]
            train_data_processor = ProcessAndTokenizeData(n_features=FEATURE_NUMS, words=train_val_words,
                                                          roots=train_val_roots,
                                                          features=train_val_features)
            categorized_features, n = train_data_processor.process_features()
            # categorized_features = pickle_handler.pickle_loader('categorized_features')
            indexed_outputs = train_data_processor.process_words_and_roots(CONTEXT_WINDOW)
            padded_indexed_outputs, max_word_len = pad_all_sequences(indexed_outputs)
            params = read_path_configs('model_params')
            model = _create_model(max_word_len, params['EMBED_DIM'], n)
            decoder_input = get_decoder_input(padded_indexed_outputs[0])
            all_inputs, all_outputs = segregate_inputs_and_outputs(padded_indexed_outputs, categorized_features,
                                                                   decoder_inputs=decoder_input)
            train_inputs, val_inputs = split_train_val(all_inputs, train_size)
            train_outputs, val_outputs = split_train_val(all_outputs, train_size)
            hist = model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs),
                             batch_size = params['BATCH_SIZE'], epochs=params['EPOCHS'],
                             callbacks=[EarlyStopping(patience=10),
                                        ModelCheckpoint('src/model_weights/model1.hdf5', save_best_only=True,
                                                        verbose=1, save_weights_only=True)
                                        ])

    else:
        print("urdu")

if __name__ == "__main__":
    main()