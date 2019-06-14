from collections import defaultdict

class DataStats(object):
    def __init__(self, sentences, features):
        self.sentences = sentences
        self.features = features

    def get_sentence_level_stats(self):
        total_len = sum([len(each) for each in self.sentences])
        total_num =len(self.sentences)
        mean_len = total_len/total_num

        return total_num, mean_len

    def get_all_words(self):
        all_words = [item for each in self.sentences for item in each]
        return all_words

    def get_tag_level_stats(self):
        unique_features = [set(each) for each in self.features]
        num_of_unique_features = [len(each) for each in unique_features]
        return num_of_unique_features

    def get_word_level_stats(self):
        all_words = self.get_all_words()
        all_features = [i for i in zip(*self.features)]
        word_feature_tuples = set([(word, feature) for word, feature in zip(all_words, all_features)])
        word_to_features_dict = defaultdict(set)
        _ = {word_to_features_dict[word].add(feature) for word, feature in word_feature_tuples}

        total_words, total_unique_words = len(all_words), len(word_to_features_dict)
        total_ambiguous_words = len([word for word in word_to_features_dict if len(word_to_features_dict[word]) > 1])
        total_unambiguous_words = total_words - total_ambiguous_words

        return total_words, total_unique_words, total_ambiguous_words, total_unambiguous_words

    def get_complete_stats(self):
        return [self.get_sentence_level_stats(), self.get_word_level_stats(), self.get_tag_level_stats()]