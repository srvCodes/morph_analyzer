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
from hindi import extract_word_root_and_feature
import yaml
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from src import handle_pickles
from copy import copy, deepcopy

parser = ArgumentParser(description="Enter --lang = 'hindi' for Hindi and 'urdu' for Urdu; "
                                    "--mode = 'train, test, or predict'")
parser.add_argument("--lang", required=True)
parser.add_argument("--mode", required=True, default='test')
args = vars(parser.parse_args())
pickle_handler= handle_pickles.PickleHandler()

LANG, MODE, = args['lang'], args['mode']
CONFIG_PATH = 'config/data_paths.yaml'

def read_path_configs():
    with open(CONFIG_PATH, 'r') as stream:
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
            return categorical_features
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



def main():
    paths = read_path_configs()
    if LANG == 'hindi':
        if MODE == 'train':
            # train_data_dir = paths['hdtb']['train']
            # train_words, train_roots, train_features = \
            #     extract_word_root_and_feature.get_words_roots_and_features(train_data_dir, n_features=8)
            # val_data_dir = paths['hdtb']['validation']
            # val_words, val_roots, val_features = \
            #     extract_word_root_and_feature.get_words_roots_and_features(val_data_dir, n_features=8)
            # assert len(train_words) == len(train_roots) == len(train_features[1]), \
            #     "Length mismatch while flattening train features"
            # assert len(val_words) == len(val_roots) == len(val_features[1]),\
            #     "Length mismatch while flattening val features"
            # # print("words: {}, roots: {}, features: {}".format(words[:5], roots[:5], features[:5]))
            # train_size, val_size = [len(each) for each in [train_words, val_words]]
            # print(train_size, val_size)
            # train_val_words, train_val_roots = [i + j for i,j in zip([train_words, train_roots], [val_words, val_roots])]
            # train_val_features = [i+j for i,j in zip(train_features, val_features)]
            # train_data_processor = ProcessAndTokenizeData(n_features=8, words=train_val_words, roots=train_val_roots,
            #                                               features=train_val_features)
            # categorized_features = train_data_processor.process_features()
            categorized_features = pickle_handler.pickle_loader('categorized_features')
            print([each[:5] for each in categorized_features])
    else:
        print("urdu")

if __name__ == "__main__":
    main()