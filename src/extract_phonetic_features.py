
class PhoneticFeatures():
    def __init__(self, list_of_words):
        self.words = list_of_words

    @staticmethod
    def place_of_articulation(word):
        labiodental = [u'\u092A', u'\u092C', u'\u092B', u'\u092D', u'\u092E', u'\u0935', ]
        dental = [u'\u091F', u'\u0920', u'\u0928', u'\u0921', u'\u0922',
                  u'\u0938', u'\u095B', u'\u0932', u'\u0931', u'\u0930']
        glottal = u'\u0939'
        is_labiodental = any((char in labiodental) for char in word)
        is_dental = any((char in dental) for char in word)
        is_glottal = any((char in glottal) for char in word)
        return [is_labiodental, is_dental, is_glottal]

    @staticmethod
    def get_svar_features(word):
        samvrit = [u'\u0907', u'\u0908', u'\u0909', u'\u090A', u'\u093F',
                   u'\u0940', u'\u0941', u'\u0942']
        ardh_samvrit = [u'\u0947', u'\u0946', u'\u094A', u'\u094B',
                        u'\u090E', u'\u090F', u'\u0913', u'\u0912']
        ardh_vivrit = [u'\u0955', u'\u0948', u'\u094C', u'\u0949',
                       u'\u0905', u'\u0910', u'\u090D', u'\u0911', u'\u0914']
        vivrit = [u'\u0906', u'\u093E']
        # lower is same as vivrit
        lower_middle = [u'\u0945', u'\u0949', u'\u0905', u'\u090D', u'\u0911']
        upper_middle = [u'\u0910', u'\u0913', u'\u0912', u'\u090E',
                        u'\u0946', u'\u0947', u'\u094A', u'\u094B']
        lower_high = [u'\u0907', u'\u0909', u'\u093F', u'\u0941']
        high = [u'\u0908', u'\u090A', u'\u0940', u'\u0942']
        total_features = [samvrit, ardh_samvrit, ardh_vivrit, vivrit,
                          lower_middle, upper_middle, lower_high, high]

        res = [any((char in item) for char in word) for item in total_features]
        return res

    @staticmethod
    def get_sthaan(word):
        y = [u'\u092A', u'\u092B', u'\u092C', u'\u092D', u'\u092E', u'\u0935']
        d = [u'\u0924', u'\u0925', u'\u0926', u'\u0927']
        v = [u'\u0928', u'\u0929', u'\u0930', u'\u0931', u'\u0933', u'\u0934', u'\u0932', u'\u0938']
        t = [u'\u091A', u'\u091B', u'\u091C', u'\u091D', u'\u091E', u'\u095F', u'\u0936', u'\u0937', u'\u092F']
        m = [u'\u091F', u'\u0920', u'\u0921', u'\u0922', u'\u0923']
        k = [u'\u0915', u'\u0916', u'\u0917', u'\u0918', u'\u0919']
        total_features = [y, d, v, t, m, k]

        res = [any((char in item) for char in word) for item in total_features]
        return res

    @staticmethod
    def get_prayatna(word):
        nasikya = [u'\u0901', u'\u0902', u'\u0903', u'\u0919', u'\u091E', u'\u0923', u'\u0928', u'\u0929', u'\u092E']
        sparsha = [u'\u0915', u'\u0916', u'\u0917', u'\u0918', u'\u091A', u'\u091B', u'\u091C', u'\u091D', u'\u091F',
                   u'\u0920', u'\u0921', u'\u0922', u'\u0924', u'\u0925', u'\u0926', u'\u0927', u'\u092A', u'\u092B',
                   u'\u092C', u'\u092D']
        parshvika = [u'\u0932', u'\u0933', u'\u0934']
        prakampi = [u'\u0930', u'\u0931']
        sangharshi = [u'\u0936', u'\u0937', u'\u0938']
        ardh_svar = [u'\u095F', u'\u092F', u'\u0935']
        total_features = [nasikya, sparsha, parshvika, prakampi, sangharshi, ardh_svar]
        res = [any((char in item) for char in word) for item in total_features]
        return res

    @staticmethod
    def vowel_types(word):
        front_vowels = [u'\u0907', u'\u0908', u'\u090F', u'\u0910']
        mid_vowels = [u'\u0905', u'\u0906']
        back_vowels = [u'\u0909', u'\u090A', u'\u0913', u'\u0914']
        long_length = [u'\u0906', u'\u0908', u'\u090A', u'\u0910', u'\u0914', u'\u093E', u'\u0940',
                       u'\u0942', u'\u0948', u'\u094C']
        short_length = [u'\u0905', u'\u0907', u'\u0909', u'\u090B', u'\u090F', u'\u093F', u'\u0941',
                        u'\u0913', u'\u094B', u'\u0947', u'\u0943']
        medium_length = [u'\u090E', u'\u090D', u'\u0912', u'\u0911', u'\u0946', u'\u0945', u'\u094A', u'\u0949']
        total_features = [front_vowels, mid_vowels, back_vowels, long_length, short_length, medium_length]
        res = [any((char in item) for char in word) for item in total_features]
        return res

    @staticmethod
    def misc_features(word):
        dravidian = [u'\u090B', u'\u0912', u'\u0931', u'\u0934', u'\u0946', u'\u094A']
        bangla = [u'\u095F']
        hard = [u'\u0937', u'\u0933', u'\u0931']
        total_features = [dravidian, bangla, hard]
        is_dravidian, is_bangla, is_hard = [any((char in item) for char in word) for item in total_features]
        return [is_dravidian, is_bangla, is_hard]

    @staticmethod
    def surface_features(word):
        vowels = [u'\u0905', u'\u0906', u'\u0907', u'\u0908', u'\u0909',
                  u'\u090A', u'\u090B', u'\u090C', u'\u090D', u'\u090E', u'\u090F',
                  u'\u0910', u'\u0911', u'\u0912', u'\u0913', u'\u0914']
        nukta = u'\u093c'
        halant = u'\u094D'
        numbers = [u'\u0966', u'\u0967', u'\u0968', u'\u0969', u'\u096A',
                   u'\u096B', u'\u096C', u'\u096D', u'\u096E', u'\u096F']
        punctuations = [u'\u0970', u'\u0971', u'\u002c', u'\u003B',
                        u'\u003f', u'\u0021', u'\u2013', u'\u002D', u'\u0022', ]
        matras = [u'\u093A', u'\u093B', u'\u093C', u'\u093D', u'\u093E', u'\u093F',
                  u'\u0940', u'\u0941', u'\u0942', u'\u0943', u'\u0944', u'\u0945',
                  u'\u0946', u'\u0947', u'\u0948', u'\u0949', u'\u094A', u'\u094B',
                  u'\u094C', u'\u094E', u'\u094F']
        voiced_aspirated = [u'\u092D', u'\u0922', u'\u0927', u'\u091D', u'\u0918', u'\u0923']
        voiceless_aspirated = [u'\u092B', u'\u0920', u'\u0925', u'\u091B', u'\u0916']
        modifiers = [u'\u0902', u'\u0901', u'\u0903']
        diphthongs = [u'\u090D', u'\u090E', u'\u090F', u'\u0910', u'\u0911', u'\u0912', u'\u0913', u'\u0914']

        total_vowels = sum([word.count(i) for i in vowels])
        nuktas = word.count(nukta)
        total_punctuations = sum([word.count(i) for i in punctuations])
        total_numbers = sum([word.count(i) for i in numbers])
        total_consonants = len(word) - (total_vowels + nuktas + total_punctuations + total_numbers)
        is_voiced_aspirated = any((char in voiced_aspirated) for char in word)
        is_voiceless_aspirated = any((char in voiceless_aspirated) for char in word)
        is_modifier = any((char in modifiers) for char in word)
        is_diphthong = any((char in diphthongs) for char in word)

        # prefix1, prefix2, prefix3 = word[:1], word[:2], word[:3]
        # suffix1, suffix2, suffix3, suffix4 = word[-1:], word[-2:], word[-3:], word[-4:]

        return [total_vowels, nuktas, total_punctuations, total_numbers, total_consonants, is_voiced_aspirated, \
               is_voiceless_aspirated, is_modifier, is_diphthong]


    def get_optimized_features_for_word(self, word):
        is_labiodental, is_dental, is_glottal = self.place_of_articulation(word)
        total_vowels, total_nuktas, total_punctuations, total_numbers, total_consonants, is_voiced_aspirated, \
        is_voiceless_aspirated, is_modifier, is_diphthong = self.surface_features(word)
        is_dravidian, is_bangla, is_hard = self.misc_features(word)
        is_front, is_mid, is_back, is_long, is_short, is_medium = self.vowel_types(word)
        is_nasikya, is_sparsha, is_parshvika, is_prakampi, is_sangarshi, is_ardhsvar = self.get_prayatna(word)
        is_dvayostha, is_dantya, is_varstya, is_talavya, is_murdhanya, is_komaltalavya = self.get_sthaan(word)
        is_samvrit, is_ardhsam, is_ardhviv, is_vivrit, is_lowmid, is_upmid, is_lowhigh, is_high = self.get_svar_features(word)

        pos_optimized = [total_punctuations, total_numbers, is_voiceless_aspirated, is_diphthong, is_glottal, is_mid,
                         is_back, is_long, is_medium, is_lowmid, is_high, is_ardhviv, is_vivrit, is_dravidian, is_komaltalavya,
                         is_sparsha,  is_prakampi, is_sangarshi]
        gen_optimized = [total_nuktas, total_vowels, total_numbers, total_consonants, is_voiceless_aspirated, is_dravidian,
                         is_mid, is_medium, is_long, is_dental, is_labiodental, is_dvayostha, is_komaltalavya, is_sparsha,
                         is_prakampi, is_sangarshi]
        num_optimized = [total_nuktas, total_punctuations, is_voiceless_aspirated, is_dravidian, is_diphthong, is_dental,
                         is_front, is_short, is_medium, is_lowhigh, is_samvrit, is_ardhsam, is_ardhviv, is_dantya, is_talavya,
                         is_nasikya, is_sparsha, is_parshvika, is_prakampi, is_ardhsvar]
        per_optimized = [total_vowels, total_nuktas, total_punctuations, total_nuktas, is_voiceless_aspirated, is_bangla,
                         is_front, is_mid, is_back, is_short, is_dental, is_glottal, is_upmid, is_lowhigh, is_high,
                         is_samvrit, is_ardhsam, is_ardhviv, is_dvayostha, is_dantya, is_varstya, is_sparsha, is_ardhsvar]
        case_optimized = [total_nuktas, total_punctuations, total_numbers, is_voiceless_aspirated, is_front, is_mid,
                          is_long, is_short, is_diphthong, is_upmid, is_lowhigh, is_dvayostha, is_varstya, is_talavya,
                          is_murdhanya, is_komaltalavya, is_nasikya, is_sparsha, is_parshvika]
        tam_optimized = [total_vowels, total_nuktas, total_numbers, is_voiceless_aspirated, is_diphthong, is_front, is_long,
                         is_medium, is_dravidian, is_bangla, is_hard, is_labiodental, is_dental, is_glottal, is_upmid,
                         is_lowhigh, is_high, is_ardhsam, is_ardhviv, is_vivrit, is_dantya, is_talavya, is_sangarshi]

        return [pos_optimized, gen_optimized, num_optimized, per_optimized, case_optimized, tam_optimized]

    def get_features(self):
        all_features = [self.get_optimized_features_for_word(word) for word in self.words]
        return all_features

if __name__ == '__main__':
    inputs = 'रेगिस्तान का मुसाफिर एक बूँद को प्यासा होता है |'
    words = inputs.split(' ')
    extractor = PhoneticFeatures(words)
    features = extractor.get_features()
    print(features)
    print(len(features), len(words))