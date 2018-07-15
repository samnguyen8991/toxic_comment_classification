# path to annotated data, used to train and test model
TRAIN_PATH = 'data/train.csv'
# path to Kaggle's unannotated test data, used for submission
TEST_PATH = 'data/test.csv'
# the number of most commonly occuring words chosen from the dataset
MAX_NUM_WORDS = 20000
# the limit length of each comment text
MAX_SEQ_LENGTH = 100
# path to pre-trained word vectors
FASTTEXT_PATH = 'data/fasttext.magnitude'
GLOVE_PATH = 'data/glove.magnitude'
EMBEDDING_DIM = 500
# path to embedding matrix
EMBEDDING_PATH = 'data/embedding_matrix.p'
# path to my trained model
MODEL_PATH = 'models/final_model.h5'
