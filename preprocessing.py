import numpy as np
import pandas as pd
from config import *

import re
from unidecode import unidecode

def clean_text(x):
    """
    Convert text to ASCII and filter special characters
    """
    x_ascii = unidecode(x)
    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]', re.IGNORECASE)
    x_clean = special_character_removal.sub('', x_ascii)
    return x_clean

def split_dataset(X_data, Y_data, train_fraction):
    """
    Split annotated data into train set and test set
    """
    split_index = int(X_data.shape[0] * train_fraction)
    X_train = X_data[:split_index]
    X_test = X_data[split_index:]
    Y_train = Y_data[:split_index, :]
    Y_test = Y_data[split_index:, :]
    return X_train, X_test, Y_train, Y_test

def load_data(args):
    """
    Load and preprocess data
    """
    train = pd.read_csv(TRAIN_PATH)
    train['clean_text'] = train['comment_text'].apply(lambda x: clean_text(str(x)))
    test = pd.read_csv(TEST_PATH)
    test['clean_text'] = test['comment_text'].apply(lambda x: clean_text(str(x)))

    x_train = train['clean_text'].fillna("something").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    test_set = test['clean_text'].fillna("something").values
    
    X_train, X_test, Y_train, Y_test = split_dataset(x_train, y_train, args.train_fraction)
    return X_train, X_test, Y_train, Y_test, test_set

