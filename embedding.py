import numpy as np
import pickle
import os
from pymagnitude import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from config import *

def embed_matrix(X_train, X_test, Test_set, args):
    """
    - Convert texts to sequences of word indices
    - Pad comment text to length of 100
    - Embed most commonly occuring words in a matrix, ith row is the embedding
    vector of word with unique index i
    @return: embedding matrix, padded train/test set
    """
    #convert raw texts to sequences of words
    #convert words to unique indices, stored in word_index file
    print('\n### TOKENIZING TEXT CORPUS ###')
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    Test_set_seq = tokenizer.texts_to_sequences(Test_set)

    x_train = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH)
    x_test = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH)
    test_set = pad_sequences(Test_set_seq, maxlen=MAX_SEQ_LENGTH)
    
    word_index = tokenizer.word_index
    print("Number of unique tokens: {}".format(len(word_index)))

    # load FastText and GLOVE pre-trained word vectors
    ft_vec = Magnitude(args.ft)
    gl_vec = Magnitude(args.gl)
    concat_vec = Magnitude(ft_vec, gl_vec)
    
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    if not os.path.isfile(EMBEDDING_PATH):
        # embed words in embedding_matrix, which concatenates FastText and Glove word vectors
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_matrix[i] = concat_vec.query(word)
        with open(EMBEDDING_PATH, 'wb') as fp:
            pickle.dump(embedding_matrix, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        #open the file with embedding matrix
        with open(EMBEDDING_PATH, 'rb') as fp:
            embedding_matrix = pickle.load(fp)

    return embedding_matrix, x_train, x_test, test_set
