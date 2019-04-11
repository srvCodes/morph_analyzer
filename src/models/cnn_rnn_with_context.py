import keras.backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Multiply, Add, Lambda, Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input, merge, \
	concatenate, GaussianNoise, dot
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder
from attention_encoder import AttentionWithContext
from keras.layers import Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm

class MorphAnalyzerModels():
    def __init__(self, all_padded_sequences, max_word_len, vocab_len, embedding_dim,
                 list_of_feature_nums, decoder_sequence, use_phonetic_features=False):
        self.all_inputs = [all_padded_sequences] + [decoder_sequence]
        self.max_len = max_word_len
        self.vocab_size = vocab_len
        self.embed_dim = embedding_dim
        self.num_filters = 64
        self.filter_len = 4
        self.hidden_dim = self.num_filters*2
        self.rnn = GRU
        self.rnn_output_size = 32
        self.dropout_rate = 0.3
        self.num_strides = 1
        self.phonetic_flag = use_phonetic_features
        self.list_of_feature_classes = list_of_feature_nums

    def apply_conv_and_pooling(self, inputs, kernel_size):
        convolutions = [Conv1D(filters=self.num_filters, kernel_size=kernel_size, padding='valid', activation='relu',
                          strides=self.num_strides, name='Conv' + str(kernel_size) + '_' + str(idx))(noise)
                        for idx, noise in enumerate(noises)]
        max_pools = [MaxPooling1D(name='MaxPool' + str(kernel_size) + '_' + str(idx))(conv) for idx, conv in
                     enumerate(convolutions)]
        avg_pools = [AveragePooling1D(name='AvgPool' + str(kernel_size) + '_' + str(idx))(conv) for idx, conv in
                     enumerate(convolutions)]
        merged_pools = [merge([i,j], name='Merge_'+str(kernel_size) + "_" + str(idx)) for idx, (i, j) in
                            enumerate(zip(max_pools, avg_pools))]
        return merged_pools


    def apply_embedding(self, input_layers, mask_flag=False):
        embedding_layers = [Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len, mask_zero=mask_flag,
                                      name='embedding_' + str(int(mask_flag)) + '_' + str(idx))(input_layer)
                            for idx, input_layer in enumerate(input_layers)]
        return embedding_layers

    def define_input_layers(self):
        input_layers = [Input(shape=self.max_len, dtype='float32', name='input' + str(idx))(sequence) for idx, sequence
                        in enumerate(self.all_inputs)]
        return input_layers

    def cnn_rnn(self):
        input_layers = self.define_input_layers()
        embedding_layers = self.apply_embedding(input_layers, mask_flag=False)
        dropouts_1 = [Dropout(self.dropout_rate, name='drop'+str(idx))(embeddings) for idx, embeddings in
                      enumerate(embedding_layers)]
        noises = [GaussianNoise(.05, name='noise'+str(idx))(dropout) for idx, dropout in enumerate(dropouts_1)]
        convolution_4 = self.apply_conv_and_pooling(inputs=noises, kernel_size=4)
        convolution_5 = self.apply_conv_and_pooling(inputs=noises, kernel_size=5)
        merge_convolutions = merge(convolution_4+convolution_5, mode='concat', name='main_merge')
        dropouts_2 = Dropout(self.dropout_rate, name='drop_1')(merge_convolutions)
        last_layer = Bidirectional(self.rnn(self.rnn_output_size), name='gru_1')(dropouts_2)
        if self.phonetic_flag is True:
            # all_features = merge([last_layer, phonetic_input], mode='concat', name='phonetic_merge')
            # dense_phonetic =  Dense(self.hidden_dim, activation='relu', kernel_initializer='he_normal', kernel_constraint=maxnorm(3),
            #             bias_constraint=maxnorm(3), name='dense_phonetic')(all_features)
            # last_layer = Dropout(self.dropout_rate, name='dropout_phonetic')(dense_phonetic)
            pass
        dense_1 = Dense(self.hidden_dim, activation='relu', kernel_initializer='he_normal', kernel_constraint=maxnorm(3),
                        bias_constraint=maxnorm(3), name='dense1')(last_layer)
        dropouts_3 = Dropout(self.dropout_rate, name='drop_2')(dense_1)
        feature_outputs = [Dense(n, kernel_initializer='he_normal', activation='softmax', name='output'+str(idx))(dropouts_3)
                            for idx, n in enumerate(list_of_feature_classes)]
        ################## seq2seq model for root prediction: Luong et. al. (2015) #####################
        encoder_embedding = self.apply_embedding([input_layers[0]], mask_flag=True)[0] # only on current word now
        encoder, state = self.rnn(self.rnn_output_size, return_sequences=True, unroll=True, return_state=True,
                             name='encoder')(encoder_embedding)
        encoder_last_state = encoder[:,-1,:]
        decoder_embedding = self.apply_embedding(input_layers[-1], mask_flag=True)
        decoder = self.rnn(self.rnn_output_size, return_sequences=True, unroll=True, name='decoder')(decoder_embedding,
                                                                                                     initial_state=
                                                                                                     [encoder_last_state])
        dot_product_1 = dot([decoder, encoder], axes=[2,2], name='dot1')
        attention = Activation('softmax', name='attention')(dot_product_1)
        dot_product_2 = dot([attention, encoder], axes=[2,1], name='dot2')
        decoder_context_combined = concatenate([dot_product_2, decoder], name='concatenate')
        outputs = TimeDistributed(Dense(self.hidden_dim/2, activation='tanh'), name='time_dist_1')(decoder_context_combined)
        output_final = TimeDistributed(Dense(self.vocab_size, activation='softmax'), name='time_dist_2')(outputs)
        ################## End of seq2seq model ###########################
        all_inputs = []
        all_outputs = [output_final]
        all_outputs += feature_outputs
        model = Model(inputs=all_inputs, outputs=all_outputs)