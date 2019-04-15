from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy

import numpy as np
import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from src import handle_pickles, process_words
from src.hindi import extract_word_root_and_feature
from src.models import cnn_rnn_with_context

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
FEATURE_NUMS = 7

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

    @staticmethod
    def get_counters_for_features(all_features, flag='original'):
        list_of_counters = [Counter(each) for each in all_features]
        if flag == 'original':
            class_labels = [list(each.keys()) for each in list_of_counters] # dump the labels to be used in filtering test set
            return class_labels
        elif flag == 'transformed':
            num_of_indiv_feature_tags = [max(each_cnt, key=int) + 1 for each_cnt in list_of_counters]
            return num_of_indiv_feature_tags

    def process_features(self):
        # list_of_counters = [Counter(each) for each in self.all_segregated_features]
        # labels = [list(each.keys()) for each in list_of_counters]
        if MODE == 'train':
            dict_of_encoders = {i: LabelEncoder() for i in range(self.n_features)}
            encoded_features = [value.fit_transform(self.all_segregated_features[idx]) for idx, value in
                                dict_of_encoders.items()]
            class_labels = self.get_counters_for_features(self.all_segregated_features, flag='original')
            num_of_indiv_feature_tags = self.get_counters_for_features(encoded_features, flag='transformed')
            categorical_features = [np_utils.to_categorical(feature, num_classes=n) for feature, n in
                                    zip(encoded_features, num_of_indiv_feature_tags)]
            _ = [pickle_handler.pickle_dumper(obj, name) for obj, name in zip([dict_of_encoders,
                                                                               num_of_indiv_feature_tags,
                                                                               categorical_features,
                                                                               class_labels],
                                                      ["dict_of_encoders", "num_of_indiv_features",
                                                       "categorized_features", 'class_labels'])]
            return categorical_features, num_of_indiv_feature_tags
        elif MODE == 'test':
            dict_of_encoders, num_of_indiv_feature_tags = [pickle_handler.pickle_loader(name) for name in
                                                           ["dict_of_encoders", "num_of_indiv_features"]]
            encoded_features_test = [dict_of_encoders[i].transform(self.all_segregated_features[i]) \
                                for i in range(self.n_features)]
            categorical_features_test = [np_utils.to_categorical(feature, num_classes=n) for feature, n in
                                         zip(encoded_features_test, num_of_indiv_feature_tags)]
            return categorical_features_test, num_of_indiv_feature_tags


    def process_words_and_roots(self, context_window=4):
        X = [item[::-1] for item in self.all_words]
        y = deepcopy(self.all_roots)
        if MODE == 'train':
            X_indexed = process_words.get_indexed_words(X, mode='build_vocab', vocab_size=VOCAB_SIZE)
        else:
            X_indexed = process_words.get_indexed_words(X, mode='use_vocab', vocab_size=VOCAB_SIZE)
        y_indexed = process_words.get_indexed_words(y, mode='use_vocab', vocab_size=VOCAB_SIZE)
        input_shifter = process_words.ShiftWordsPerCW(X=X, cw=context_window, vocab_size=VOCAB_SIZE)
        X_indexed_left, X_indexed_right = input_shifter.shift_input()
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
    max_word_len = max(max([len(word) for word in indexed_outputs[0]]), max([len(word) for word in indexed_outputs[-1]]))
    all_padded_inputs = [sequence_padder(each, max_word_len) for each in indexed_outputs]
    return all_padded_inputs, max_word_len


def _create_model(max_word_len, embed_dim, n):
    model_instance = cnn_rnn_with_context.MorphAnalyzerModels(max_word_len=max_word_len, vocab_len=VOCAB_SIZE+2,
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


def segregate_inputs_and_outputs(words_and_roots, features, decoder_inputs):
    roots = words_and_roots[-1]
    inputs = words_and_roots[:-1]
    inputs.append(decoder_inputs)
    outputs = [roots]
    outputs += features
    return inputs, outputs


def write_features_to_file(words, orig_features, pred_features, output_path):
    encoders = pickle_handler.pickle_loader('dict_of_encoders')
    orig_features = [[np.where(idx==1)[0][0] for idx in each] for each in orig_features]
    pred_features = [each.tolist() for each in pred_features]
    orig_transformed_features = [encoders[i].inverse_transform(orig_features[i]) for i in range(FEATURE_NUMS)]
    pred_transformed_features = [encoders[i].inverse_transform(pred_features[i]) for i in range(FEATURE_NUMS)]
    for idx in range(FEATURE_NUMS):
        filename = output_path+'feature_'+str(idx)+'.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Word\t\tOriginal_feature\t\tPredicted_feature\n")
            for i,j,k in zip(words, orig_transformed_features[idx], pred_transformed_features[idx]):
                f.write(i + '\t\t' + str(j) + '\t\t' + str(k) + '\n')
            f.close()


def write_roots_to_file(words, orig_roots, pred_roots, output_path):
    idx_to_char_mapping = pickle_handler.pickle_loader('index_to_char_mapping')
    pred_sequences = list()
    for each in pred_roots:
        list_of_chars = list()
        list_of_chars += [idx_to_char_mapping[idx] for idx in each if idx > 0]
        sequence = ''.join(list_of_chars)
        pred_sequences.append(sequence)
    out_file = output_path+'_words.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("Word\t\tOriginal_root\t\tPredicted_root\n")
        for i,j,k in zip(words, orig_roots, pred_sequences):
            f.write(i + '\t\t' + j + '\t\t' + str(k) + '\n')
        f.close()


class RemoveErroneousIndices():
    def __init__(self, test_file_contents):
        self.contents = test_file_contents
        self.class_labels = pickle_handler.pickle_loader('class_labels')
        self.erroneous_indices = self.get_erroneous_indices()

    def filter_erroneous_indices(self, _list):
        # print("indices: ", self.erroneous_indices)
        _list = [each for i, each in enumerate(_list) if i not in self.erroneous_indices]
        return _list


    def get_erroneous_indices(self):
        erroneous_indices = list()
        features = self.contents[-1]
        for i, (feature, label) in enumerate(zip(features, self.class_labels)):
            for idx in range(len(feature)):
                print("{} {}".format(feature[idx], label))
                if feature[idx] not in label or (i == 6 and feature[idx] == 'kI'):
                    erroneous_indices.append(idx)
        return erroneous_indices


    def remove_unknown_feature_labels(self): # file_contents = [words, roots, features]
        words, roots = [self.filter_erroneous_indices(each) for each in self.contents[:2]]
        features = [self.filter_erroneous_indices(each) for each in self.contents[-1]]
        return words, roots, features


class ProcessDataForModel():
    def __init__(self, words, roots, features):
        self.words = words
        self.roots = roots
        self.features = features

    def process_end_to_end(self):
        data_processor = ProcessAndTokenizeData(n_features=FEATURE_NUMS, words=self.words,
                                                roots=self.roots,
                                                features = self.features)
        categorized_features, n = data_processor.process_features()
        indexed_inputs = data_processor.process_words_and_roots(CONTEXT_WINDOW)
        padded_indexed_inputs, max_word_len = pad_all_sequences(indexed_inputs)
        padded_indexed_inputs[-1] = process_words.one_hot_encode_output_data(
            padded_indexed_inputs[-1], max_word_len, VOCAB_SIZE+2
        )
        decoder_input = get_decoder_input(padded_indexed_inputs[0])
        all_inputs, all_outputs = segregate_inputs_and_outputs(padded_indexed_inputs, categorized_features,
                                                               decoder_inputs=decoder_input)
        return [all_inputs, all_outputs, max_word_len, n]



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
            # print(len(train_words), len(train_roots), len(train_features[0]))
            # print("words: {}, roots: {}, features: {}".format(train_words[:5], train_roots[:5], train_features[:5]))
            train_size, val_size = [len(each) for each in [train_words, val_words]]
            train_val_words, train_val_roots = [train_words + val_words, train_roots + val_roots]
            train_val_features = [i+j for i,j in zip(train_features, val_features)]
            train_data_generator = ProcessDataForModel(words=train_val_words, roots=train_val_roots,
                                                       features=train_val_features)
            all_inputs, all_outputs, max_word_len, n = train_data_generator.process_end_to_end()
            params = read_path_configs('model_params.yaml')
            model = _create_model(max_word_len, params['EMBED_DIM'], n)
            train_inputs, val_inputs = split_train_val(all_inputs, train_size)
            train_outputs, val_outputs = split_train_val(all_outputs, train_size)
            hist = model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs),
                             batch_size = params['BATCH_SIZE'], epochs=params['EPOCHS'],
                             callbacks=[EarlyStopping(patience=10),
                                        ModelCheckpoint(paths['model_weights'], save_best_only=True,
                                                        verbose=1, save_weights_only=True)
                                        ])
        elif MODE == 'test':
            test_data_dir = paths['hdtb']['test']
            contents = extract_word_root_and_feature.get_words_roots_and_features(
                test_data_dir, n_features=FEATURE_NUMS
            )
            index_identifier = RemoveErroneousIndices(contents)
            test_words, test_roots, test_features = index_identifier.remove_unknown_feature_labels()
            test_data_generator = ProcessDataForModel(words=test_words, roots=test_roots,
                                                       features=test_features)
            all_inputs, all_outputs, max_word_len, n = test_data_generator.process_end_to_end()
            params = read_path_configs('model_params.yaml')
            model = _create_model(max_word_len, params['EMBED_DIM'], n)
            model.load_weights(paths['model_weights'])
            pred_outputs = model.predict(all_inputs)
            predicted_char_indices = np.argmax(pred_outputs[0], axis=2)
            predicted_features = [np.argmax(each, axis=1) for each in pred_outputs[1:]]
            _ = write_features_to_file(test_words, all_outputs[1:], predicted_features, paths['output'])
            _ = write_roots_to_file(test_words, test_roots, predicted_char_indices, paths['output'])
        elif MODE == 'predict':
            test_data_dir = paths['hindi_test']
            sentences = extract_word_root_and_feature.get_words_for_predictions(test_data_dir)
            predictions = list()
            for sentence in sentences:
                words_reversed = [item[::-1] for item in sentence]
                X_indexed = process_words.get_indexed_words(words_reversed, mode='use_vocab', vocab_size=VOCAB_SIZE)
                input_shifter = process_words.ShiftWordsPerCW(X=words_reversed, cw=CONTEXT_WINDOW, vocab_size=VOCAB_SIZE)
                X_indexed_left, X_indexed_right = input_shifter.shift_input()
                all_inputs = list()
                all_inputs.append(X_indexed)
                all_inputs += X_indexed_left
                all_inputs += X_indexed_right
                padded_indexed_inputs, max_word_len = pad_all_sequences(all_inputs)
                n = pickle_handler.pickle_loader('num_of_indiv_features')
                decoder_input = get_decoder_input(padded_indexed_inputs[0])
                all_inputs.append(decoder_input)
                params = read_path_configs('model_params.yaml')
                model = _create_model(max_word_len, params['EMBED_DIM'], n)
                model.load_weights(paths['model_weights'])
                pred_outputs = model.predict(all_inputs)
                predictions.append(pred_outputs)
                print(predictions)
            _ = write_predicted_roots_and_features(sentences, predictions)
    else:
        print("urdu")

if __name__ == "__main__":
    main()