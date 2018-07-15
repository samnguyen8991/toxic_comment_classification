from config import *

from keras.models import Model
from keras.layers import Input, Embedding, Dense, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import load_model

class CommentClassifier:
    def __init__(self, args):
        self.args = args 
        self.neural_net = None 

    def build_model(self, embedding_matrix):
        # Layer 1: Embedding layer
        seq_input = Input(shape=(MAX_SEQ_LENGTH,))
        x = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix],
                                    input_length=MAX_SEQ_LENGTH, trainable=False)(seq_input)

        # Layer 2: SpatialDropout1D, dropout rate 0.5 (default)
        x = SpatialDropout1D(self.args.dropout)(x)

        # Layer 3: Bidirectional LSTM, each memory cell has 60 units (default)
        x = Bidirectional(LSTM(self.args.lstm_unit, return_sequences=True))(x)

        # Layer 4: Bidirectional LSTM, each memory cell has 60 units (default)
        x = Bidirectional(LSTM(self.args.lstm_unit, return_sequences=True))(x)

        # Layer 5: Bidirectional GRU, each memory cell has 60 units (default)
        x, x_last, x_c = Bidirectional(GRU(self.args.gru_unit, return_sequences=True, return_state=True))(x)

        #Layer 6: concatenate avg_pool, output of last hidden state, max_pool
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, x_last, max_pool])

        # Layer 7: Dense layer, dim of output = 6
        preds = Dense(6, activation="sigmoid")(x)

        self.neural_net = Model(inputs=seq_input, outputs=preds)
        self.neural_net.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        print("\n### MODEL ARCHITECTURE ###")
        self.neural_net.summary()
        print()
    
    def train(self, x_train, Y_train, x_test, Y_test):
        print("\n### TRAINING ###")
        history = self.neural_net.fit(x_train, Y_train, validation_data=(x_test, Y_test), batch_size=self.args.batch_size, epochs=self.args.n_epochs, verbose=self.args.verbose, shuffle=True)
        self.neural_net.save(self.args.save)

    def load_model(self):
        self.neural_net = load_model('models/final_model.h5')

    def predict(self, x):
        return self.neural_net.predict(x)
    
    def evaluate(self, x_test, y_test):
        return self.neural_net.evaluate(x_test, y_test, verbose=0)
