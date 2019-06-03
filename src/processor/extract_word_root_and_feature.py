import os
import re
from copy import deepcopy

class ParseFile():
    def __init__(self, path):
        self.path = path
        self.sentences_with_words = list()
        self.sentences_with_roots = list()
        self.sentences_with_features = list()


    @staticmethod
    def read_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[:40]


    def get_content_from_all_lines(self, lines):
        words, roots, features = [], [], []
        for line in lines:
            line = line.strip()
            if len(line) > 0:    # keep adding words till blank line
                entities = re.split(r'\t+', line.rstrip('\t'))
                words.insert(len(words), entities[1])
                roots.insert(len(roots), entities[2])
                features.insert(len(features), entities[5])
                continue
            else:   # on encountering a blank line, add all previous words to form a sentence
                _words, _roots, _features = [deepcopy(each) for each in [words, roots, features]]
                self.sentences_with_words.append(_words)
                self.sentences_with_roots.append(_roots)
                self.sentences_with_features.append(_features)
                words.clear()
                roots.clear()
                features.clear()


    def flatten_and_segregate_features(self, n_features):
        flat_features = [item for sublist in self.sentences_with_features for item in sublist]
        splitted_features = [feature.split('|') for feature in flat_features]
        all_features = [[], [], [], [], [], [], []]
        for feature in splitted_features:
            for idx, (i,j) in enumerate(zip(feature[:n_features+1], all_features[:n_features+1])):
                val = re.sub(r'.*-', '', i)
                _ = [j.append(val) if len(val) > 0 else j.append('UNK')]
        del all_features[5]
        return all_features


    def flatten_words_and_roots(self):
        _all_words, _all_roots = [[item for sentence in sentences for item in sentence] for sentences in \
                                  [self.sentences_with_words, self.sentences_with_roots]]
        return _all_words, _all_roots


    def read_dir(self, lang='hindi'):
        cnt = 0
        for item in os.listdir(self.path):
            if lang == 'hindi':
                filepath = os.path.join(self.path, item)
                lines = self.read_file(filepath)
                self.get_content_from_all_lines(lines)
            elif lang == 'urdu':
                for file in os.listdir(os.path.join(self.path, item)):
                    filepath = os.path.join(*[self.path, item, file])
                    lines = self.read_file(filepath)
                    self.get_content_from_all_lines(lines)
        return self.sentences_with_words, self.sentences_with_roots, self.sentences_with_features


def get_words_roots_and_features(path, n_features, lang='hindi'):
    file_parser = ParseFile(path)
    _, _, _ = file_parser.read_dir(lang=lang)
    indiv_features = file_parser.flatten_and_segregate_features(n_features=n_features)
    all_words, all_roots = file_parser.flatten_words_and_roots()
    return all_words, all_roots, indiv_features

def get_words_for_predictions(data_dir):
    sentences = [line.split() for line in open(data_dir, 'r', encoding='utf-8').readlines()]
    return sentences

if __name__ == "__main__":
    pass
