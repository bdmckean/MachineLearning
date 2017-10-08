import os
import json
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features

# TODO: Add custom feature transformers for the movie review data
class TextPosWordTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        good_words = ['clever', 'riveting', 'best', 'oscar', 'enjoyable', 'charming', 'absorbing', 'powerful', 'dazzling']
        features = np.zeros((len(examples), 1))
        for idx, ex in enumerate(examples):
            features[idx, 0] = len([ x for x in good_words if x in ex] )
        #print ("Pos word features", features)
        return features


# TODO: Add custom feature transformers for the movie review data
class TextNegWordTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        good_words = ['moronic', 'boring', 'bloody', 'disgusting', 'flawed', 'predicable', 'senseless', 'weak', 'uneven']
        features = np.zeros((len(examples), 1))
        for idx, ex in enumerate(examples):
            features[idx, 0] = len([ x for x in good_words if x in ex] )
        #print ("Neg word features", features)
        return features


class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf_vectorizer =  TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        self.tranformer = None

    def fit(self, examples):
        #print("tfidf fit", examples[:1])
        self.transformer = self.tfidf_vectorizer.fit(examples)
        return self

    def transform(self, examples):
        #print("tfidf transform", examples[:1])
        features = None
        features = self.transformer.transform(examples)
        #print (features[0])
        return features

class CountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vectorizer =  CountVectorizer(analyzer='word', lowercase=True, ngram_range=(2, 3), stop_words='english')
        self.tranformer = None

    def fit(self, examples):
        #print("count fit", examples[:1])
        self.transformer = self.count_vectorizer.fit(examples)
        return self

    def transform(self, examples):
        #print("count transform", examples[:1])
        features = None
        features = self.transformer.transform(examples)
        #print (features[0])
        return features


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            #('text_stats', Pipeline([
            #    ('selector', ItemSelector(key='text')),
            #    ('text_length', TextLengthTransformer())
            #]))
            #,
            ('text_stats2', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfTransformer())
            ]))
            ,
            ('text_stats3', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('TextNegWordTransformer', TextNegWordTransformer())
            ]))
            ,
            ('text_stats4', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('TextPosWordTransformer', TextPosWordTransformer())
            ]))
            ,
            ('text_stats5', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('CountTransformer', CountTransformer())
            ]))
        ])  

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))
    #exit(0)
    # Train classifier
    #lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=1500, shuffle=True, verbose=0)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)
    exit(0)

    # Code for extra credirs
    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier
    N_FEATURES_OPTIONS = [2, 4, 8]
    ALPHA_OPTIONS = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    param_grid = [
        {
            'alpha': ALPHA_OPTIONS
        }
        ]
    
    for alpha in ALPHA_OPTIONS:
        print( "Alpha= ", alpha)
        lr = SGDClassifier(loss='log', penalty='l2', alpha=alpha, max_iter=1500, shuffle=True, verbose=0)
        lr.fit(feat_train, y_train)
        y_pred = lr.predict(feat_train)
        accuracy = accuracy_score(y_pred, y_train)
        print("Accuracy on training set =", accuracy)
        y_pred = lr.predict(feat_test)
        accuracy = accuracy_score(y_pred, y_test)
        print("Accuracy on test set =", accuracy)

        scores = cross_val_score(lr, feat_train, y_train, cv=10)
        print("Avg Score=", sum(scores)/len(scores), scores)

      
