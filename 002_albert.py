import argparse
import os

import numpy as np
from sklearn.cross_validation import StratifiedKFold

import autosklearn
import autosklearn.data
import autosklearn.data.competition_data_manager
from autosklearn.evaluation.util import calculate_score
from ParamSklearn.classification import ParamSklearnClassifier


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

input = args.input
dataset = 'albert'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(1.0, ParamSklearnClassifier(configuration={
        'balancing:strategy': 'weighting',
        'classifier:__choice__': 'sgd',
        'classifier:sgd:loss': 'hinge',
        'classifier:sgd:penalty': 'l2',
        'classifier:sgd:alpha': 0.0001,
        'classifier:sgd:fit_intercept': True,
        'classifier:sgd:n_iter': 5,
        'classifier:sgd:learning_rate': 'optimal',
        'classifier:sgd:eta0': 0.01,
        'classifier:sgd:average': True,
        'imputation:strategy': 'mean',
        'one_hot_encoding:use_minimum_fraction': 'True',
        'one_hot_encoding:minimum_fraction': 0.1,
        'preprocessor:__choice__': 'no_preprocessing',
        'rescaling:__choice__': 'min/max'}))]

classifiers = []
targets = []
predictions = []
predictions_valid = []
predictions_test = []

# Make predictions and weight them
iteration = 0
for weight, classifier in choices:
    iteration += 1
    print dataset, "Iteration %d/%d" % (iteration, len(choices))
    classifiers.append(classifier)
    try:
        classifier.fit(X.copy(), y.copy())
        predictions_valid.append(
            classifier.predict_proba(X_valid.copy()) * weight)
        predictions_test.append(
            classifier.predict_proba(X_test.copy()) * weight)
    except Exception as e:
        print e
        print classifier.configuration

# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0)
    predictions = predictions[:, 1].reshape((-1, 1))

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ')
