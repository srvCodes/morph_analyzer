import os
from nltk import FreqDist
import numpy as np
import re
import datetime
import sys
import gc
import pickle

from collections import Counter, deque
from argparse import ArgumentParser
from hindi import extract_word_root_and_feature
import yaml

parser = ArgumentParser(description="Enter --lang = 'hindi' for Hindi and 'urdu' for Urdu; --mode = 'train, test, or predict'")
parser.add_argument("--lang", required=True)
parser.add_argument("--mode", required=True, default='test')
args = vars(parser.parse_args())

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
    def __init__(self, path, n_features):
        self.data_dir = path
        self.n_features = n_features
        self.all_words, self.all_roots, self.all_features = self.get_words_roots_and_features()


    def get_words_roots_and_features(self):
        words, roots, features = \
            extract_word_root_and_feature.get_words_roots_and_features(self.data_dir, n_features=self.n_features)
        return words, roots, features

    def process_features(self):
        
def main():
    paths = read_path_configs()
    if LANG == 'hindi':
        if MODE == 'train':
            train_data_dir = paths['hdtb']['train']
            train_data_processor = ProcessAndTokenizeData(train_data_dir, n_features=8)
            val_data_dir = paths['hdtb']['validation']
            val_data_processor = ProcessAndTokenizeData(val_data_dir)

            val_words, val_roots, val_features = \
                extract_word_root_and_feature.get_words_roots_and_features(val_data_dir, n_features=8)
            assert len(train_words) == len(train_roots) == len(train_features[1]), \
                "Length mismatch while flattening train features"
            assert len(val_words) == len(val_roots) == len(val_features[1]),\
                "Length mismatch while flattening val features"
            # print("words: {}, roots: {}, features: {}".format(words[:5], roots[:5], features[:5]))
    else:
        print("urdu")

if __name__ == "__main__":
    main()