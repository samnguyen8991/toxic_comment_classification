"""
Reference
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
"""

import numpy as np
import pickle
import os
from pymagnitude import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from config import *

def embed_matrix(X_train, X_test, Test_set, args):
    """
    - Map words to continuous, real-valued embedding vectors
    - Convert texts to sequences of word indices
    - Embed most commonly occuring words in a matrix, ith row is the embedding
    vector of word with unique index i
    - Pad comment text to length of 100
    Returns the embedding matrix, x_train, x_test, test_set (padded versions)
    """
    #convert raw texts to sequences of words
    #convert words to unique indices, stored in word_index file
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    Test_set_seq = tokenizer.texts_to_sequences(Test_set)

    x_train = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH)
    x_test = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH)
    test_set = pad_sequences(Test_set_seq, maxlen=MAX_SEQ_LENGTH)

    # word_index stores the word-to-index mapping
    if not os.path.isfile(WORD_INDEX_PATH):
        print('\n### TOKENIZING TEXT CORPUS ###')
        word_index = tokenizer.word_index
        # save word_index of most common words to a file
        with open(WORD_INDEX_PATH, 'wb') as fp:
            pickle.dump(word_index, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(WORD_INDEX_PATH, 'rb') as fp:
            word_index = pickle.load(fp)
        print("Number of unique tokens: {}".format(len(word_index)))

    # load FastText and GLOVE pre-trained word vectors
    ft_vec = Magnitude(args.ft)
    gl_vec = Magnitude(args.gl)
    concat_vec = Magnitude(ft_vec, gl_vec)

    """
    uncomment if you want to recreate the dictionaries that map word to its word vectors
    """
    # load pre-trained  Fasttext word vectors into word_vectors_ft dictionary
    # with open(FASTTEXT_WORD_VECTORS, encoding='utf-8') as f:
    #     for line in f:
    #         word = line.rstrip().rsplit(' ')
    #         try:
    #             word_vectors_ft[word[0]] = np.asarray(word[1:], dtype='float32')
    #         except ValueError:
    #             continue
    #
    # #load pre-trained GLOVE word vectors into word_vectors_tw dictionary
    # with open(GLOVE_WORD_VECTORS, encoding='utf-8') as f:
    #     for line in f:
    #         word = line.rstrip().rsplit(' ')
    #         try:
    #             word_vectors_tw[word[0]] = np.asarray(word[1:], dtype='float32')
    #         except ValueError:
    #             continue
    #
    # # save dictionaries to a file
    # with open('/scratch/mnguyen7/data/word_vectors_ft.p', 'wb') as fp:
    #     pickle.dump(word_vectors_ft, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/scratch/mnguyen7/data/word_vectors_tw.p', 'wb') as fp:
    #     pickle.dump(word_vectors_tw, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #open the file with dictionaries
    #with open('/scratch/mnguyen7/data/word_vectors_ft.p', 'rb') as fp:
        #word_vectors_ft = pickle.load(fp)
    #with open('/scratch/mnguyen7/data/word_vectors_tw.p', 'rb') as fp:
        #word_vectors_tw = pickle.load(fp)
    #print("Shape of each Fasttext word vector: {}".format(word_vectors_ft["something"].shape))
    #print("Shape of each GLOVE word vector: {}".format(word_vectors_tw["something"].shape))

    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    """
    uncomment if you want to recreate the embedding matrix
    this is the matrix we use for our final model
    """
        # embedding_vector_ft = word_vectors_ft.get(word)
        # if embedding_vector_ft is not None:
            #embedding_matrix[i,:300] = embedding_vector_ft
        #embedding_vector_tw = word_vectors_tw.get(word)
        #if embedding_vector_tw is not None:
            #embedding_matrix[i,300:] = embedding_vector_tw
        #if embedding_vector_ft is None and embedding_vector_tw is None:
             #embedding_matrix[i,:300] = word_vectors_ft["something"]
             #embedding_matrix[i,300:] = word_vectors_tw["something"]
    
    if not os.path.isfile(EMBEDDING_PATH):
        # embed words in embedding_matrix, which concatenates FastText and Glove word vectors
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_matrix[i] = concat_vec.query(word)
        print('should not be here')
        with open(EMBEDDING_PATH, 'wb') as fp:
            pickle.dump(embedding_matrix, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #open the file with embedding matrix
    with open(EMBEDDING_PATH, 'rb') as fp:
        embedding_matrix = pickle.load(fp)

    return embedding_matrix, x_train, x_test, test_set