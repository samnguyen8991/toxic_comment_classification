# Toxic Comment Classification 
Build a multi-headed RNN model that detects and classify six types of toxic comments: toxic, severe toxic, threats, obscene, insult and identity hate. Model is trained with a dataset of comments on Wikipedia's talk page edits. 

Test accuracy: 98.4%  
AUC score: 0.987

## Dependencies 
* Python3 
* Pandas 
* Tensorflow + Keras 
* Scikit-learn 
* [Magnitude](https://github.com/plasticityai/magnitude) (store and concatenate word vectors)
* Unidecode (preprocess text)

## Dataset 
* train.csv.zip: a dataset of comment texts, each annotated with 6 binary labels 
* test.csv.zip: a dataset of unannotated comment texts, provided on Kaggle for competition submission. This file is included for text tokenization only and will not be used as test set. 

Unzip the datasets by running:
```bash
unzip <filename>
```

## Word embeddings 
This model uses both FastText and Glove pre-trained word vectors for its Embedding layer. A package called Magnitude is utilized to efficiently store, normalize and concatenate vectors from the two embedding models. You can visit [Magnitude](https://github.com/plasticityai/magnitude) to read more about mechanisms to handle out-of-vocabulary words. Special thanks to this package for significantly improving the performance of my model!

Download pre-trained word vectors in Magnitude format by running:
```bash
# FastText: 300-dim word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)
curl http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M-subword.magnitude --output data/fasttext.magnitude
# Glove: 200-dim word vectors trained on Wikipedia 2014 and Gigaword 5 (6B tokens)
curl http://magnitude.plasticity.ai/glove+subword/glove.6B.200d.magnitude --output data/glove.magnitude
```

## Training 
Run the following command to train a model: 
```bash
python3 classify.py --save <model_path> 
(e.g python3 classify.py --save models/model.h5)
```

Run the following command to load a saved model: 
```bash
python3 classify.py --load <model_path>
```
If both '--save' and '--load' are not specified, my trained model, stored at 'models/final_model.h5' will be loaded by default. If you're unable to load the model, please try to upgrade Keras to the latest version:
```bash
python3 -m pip install keras --upgrade
```

Run the following command to check all the parameters you can tune: 
```bash
python3 classify.py -h
```

## Reference 
Ideas for model architecture are inspired by https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644
