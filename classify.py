import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score

from preprocessing import load_data
from embedding import embed_matrix
from neural_net import CommentClassifier
from config import *

def parse_args():
    """
    Parse parameters
    """
    parser = ArgumentParser()
    # preprocessing and embedding 
    parser.add_argument('--t', type=float, default=0.8, help="fraction of data for testing")
    parser.add_argument('--ft', type=str, default=FASTTEXT_PATH, help='path of FastText word vectors')
    parser.add_argument('-gl', type=str, default=GLOVE_PATH, help='path of Glove word vectors')
    
    # model architecture
    parser.add_argument('--lstm_unit', type=str, default=60, help='number of hidden units for LSTM layer')
    parser.add_argument('--gru_unit', type=str, default=60, help='number of hidden units for GRU layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='spatial dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--verbose', type=int, default=1, help='verbose: 0, 1, 2')
    parser.add_argument('--save', type=str, help='path to save models')
    parser.add_argument('--load', type=str, help='path to load models')
    
    args = parser.parse_args()
    return args
 
def main():
    args = parse_args()
    X_train, X_test, Y_train, Y_test, Test_set = load_data(args)
    print('### PREPROCESSING AND LOADING DATA ###')
    print ("Shape of x_train: {}, shape of x_test: {}".format(X_train.shape, X_test.shape))
    print ("Shape of y_train: {}, shape of y_test: {}".format(Y_train.shape, Y_test.shape))

    embedding_matrix, x_train, x_test, test_set = embed_matrix(X_train, X_test, Test_set, args)
    print("Shape of embedding matrix: {}".format(embedding_matrix.shape))

    # train or load model
    if args.save == None and args.load == None:
        raise Exception('usage: --save: path to save a newly-trained model; --load: path to load an old model')
    neural_net = CommentClassifier(args)
    if args.save:
        neural_net.build_model(embedding_matrix)
        neural_net.train(x_train, Y_train, x_test, Y_test)
    if args.load:
        neural_net.load_model()

    # evaluate model, metrics: binary accuracy
    loss, accuracy = neural_net.evaluate(x_test, Y_test, verbose=0)
    print("\n### EVALUATING ###")
    print("Test accuracy: {}".format(accuracy*100))

    # evaluate model, metrics: AUC score
    y_pred = neural_net.predict(x_test)
    roc_auc = roc_auc_score(Y_test, y_pred)
    print("ROC-AUC score: {}".format(roc_auc))

if __name__ == "__main__":
    main()
 
