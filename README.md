# Toxic Comment Classification 
Build a multi-headed model that detects and classify six types of toxic comments: toxic, severe toxic, threats, obscene, insult and identity hate. Model is trained with a dataset of comments on Wikipedia's talk page edits. 

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
This model uses both FastText and Glove pre-trained word vectors for its Embedding layer. A package called Magnitude is utilized to efficiently store, normalize and concatenate vectors from the two embedding models. You can visit [Magnitude](https://github.com/plasticityai/magnitude) to read more about mechanisms to handle out-of-vocabulary words. 

Download pre-trained word vectors in Magnitude format by running:
```bash
# FastText: 300-dim word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)
curl http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M-subword.magnitude --output data/fasttext.magnitude
# Glove: 200-dim word word vectors trained on Wikipedia 2014 and Gigaword 5 (6B tokens)
curl http://magnitude.plasticity.ai/glove+subword/glove.6B.200d.magnitude --output data/glove.magnitude
```

# Training 
Run the following command to train a model: 
```bash
python3 classify.py --save <model_path> 
(e.g python3 classify.py --save models/final_model.h5)
```

Run the following command to load a saved model: 
```bash
python3 classify.py --load <model_path>
```

Run the following command to check all the parameters you can tune: 
```bash
python3 classify.py -h
```
