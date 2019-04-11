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
    def __init__(self, all_padded_sequences, max_word_len):
        self.all_inputs = all_padded_sequences
        self.max_len = max_word_len

    def cnn_rnn(self):
        input_layers = [Input(shape=self.max_len, dtype='float32', name='input' + str(i))(sequence) for idx, sequence
                        in enumerate]