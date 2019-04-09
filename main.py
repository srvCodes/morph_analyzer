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

parser = ArgumentParser(description="Enter --lang = 'hindi' for Hindi and 'urdu' for Urdu")
parser.add_argument("--lang", required=True)
args = vars(parser.parse_args())

CONFIG_PATH = 'config/data_paths.yaml'

def read_path_configs():
    with open(CONFIG_PATH, 'r') as stream:
        try:
            res = yaml.load(stream)
        except yaml.YAMLError as e:
            res = None
            print("Error while reading yaml: ", e)
    return res


def main():
    paths = read_path_configs()
    if args['lang'] == 'hindi':
        data_dir = paths['hdtb']
        words, roots, features = extract_word_root_and_feature.get_words_roots_and_features(data_dir, n_features=8)
        print(len(words), len(roots), len(features[1]))
        # print("words: {}, roots: {}, features: {}".format(words[:5], roots[:5], features[:5]))
    else:
        print("urdu")

if __name__ == "__main__":
    main()