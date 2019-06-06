class DataStats(object):
    def __init__(self, sentences, features):
        self.sentences = sentences
        self.features = features

    def get_sentence_level_stats(self):
        total_len = sum([len(each) for each in self.sentences])
        total_num =len(self.sentences)
        mean_len = total_len/total_num

        return total_num, mean_len


    def get_word_level_stats(self):
        all_words = [item for each in self.sentences for item in each]
        unique_words = set(all_words)

        return len(all_words), len(unique_words)

    def get_tag_level_stats(self):
        unique_features = [set(each) for each in self.features]
        num_of_unique_features = [len(each) for each in unique_features]
        return num_of_unique_features

    def get_complete_stats(self):
        return [self.get_sentence_level_stats(), self.get_word_level_stats(), self.get_tag_level_stats()]