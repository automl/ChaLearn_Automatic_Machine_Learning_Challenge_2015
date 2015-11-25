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
dataset = 'dilbert'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.220000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'weighting',
            'classifier:__choice__': 'passive_aggressive',
            'classifier:passive_aggressive:C': 0.0022574783522003694,
            'classifier:passive_aggressive:fit_intercept': 'True',
            'classifier:passive_aggressive:loss': 'hinge',
            'classifier:passive_aggressive:n_iter': 119,
            'imputation:strategy': 'most_frequent',
            'one_hot_encoding:minimum_fraction': 0.1898871876010834,
            'one_hot_encoding:use_minimum_fraction': 'True',
            'preprocessor:__choice__': 'gem',
            'preprocessor:gem:N': 20,
            'preprocessor:gem:precond': 0.27540716190663134,
            'rescaling:__choice__': 'min/max'})),
     (0.160000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'none',
            'classifier:__choice__': 'passive_aggressive',
            'classifier:passive_aggressive:C': 8.011168723835382,
            'classifier:passive_aggressive:fit_intercept': 'True',
            'classifier:passive_aggressive:loss': 'hinge',
            'classifier:passive_aggressive:n_iter': 20,
            'imputation:strategy': 'median',
            'one_hot_encoding:minimum_fraction': 0.020771877142610626,
            'one_hot_encoding:use_minimum_fraction': 'True',
            'preprocessor:__choice__': 'gem',
            'preprocessor:gem:N': 16,
            'preprocessor:gem:precond': 0.035878450355803344,
            'rescaling:__choice__': 'min/max'})),
     (0.160000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 0.00010934133255683256,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'hinge',
             'classifier:passive_aggressive:n_iter': 235,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.022038507512545786,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 17,
             'preprocessor:gem:precond': 0.02104468261583234,
             'rescaling:__choice__': 'min/max'})),
     (0.140000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 8.011168723835382,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'hinge',
             'classifier:passive_aggressive:n_iter': 20,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.020771877142610626,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 16,
             'preprocessor:gem:precond': 0.047677121638912856,
             'rescaling:__choice__': 'min/max'})),
     (0.140000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 8.011168723835382,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'squared_hinge',
             'classifier:passive_aggressive:n_iter': 301,
             'imputation:strategy': 'median',
             'one_hot_encoding:minimum_fraction': 0.028040769173853935,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 20,
             'preprocessor:gem:precond': 0.047677121638912856,
             'rescaling:__choice__': 'min/max'})),
     (0.120000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 0.00010934133255683256,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'hinge',
             'classifier:passive_aggressive:n_iter': 235,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.041303833357502165,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 18,
             'preprocessor:gem:precond': 0.09599232591423834,
             'rescaling:__choice__': 'min/max'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'liblinear_svc',
             'classifier:liblinear_svc:C': 37.176582995422606,
             'classifier:liblinear_svc:dual': 'False',
             'classifier:liblinear_svc:fit_intercept': 'True',
             'classifier:liblinear_svc:intercept_scaling': 1,
             'classifier:liblinear_svc:loss': 'squared_hinge',
             'classifier:liblinear_svc:multi_class': 'ovr',
             'classifier:liblinear_svc:penalty': 'l2',
             'classifier:liblinear_svc:tol': 0.00016373824508657717,
             'imputation:strategy': 'median',
             'one_hot_encoding:minimum_fraction': 0.0008207509562933506,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 15,
             'preprocessor:gem:precond': 0.1010713117945701,
             'rescaling:__choice__': 'min/max'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 8.011168723835382,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'squared_hinge',
             'classifier:passive_aggressive:n_iter': 20,
             'imputation:strategy': 'median',
             'one_hot_encoding:minimum_fraction': 0.028040769173853935,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'gem',
             'preprocessor:gem:N': 20,
             'preprocessor:gem:precond': 0.047677121638912856,
             'rescaling:__choice__': 'min/max'}))
    ]

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
        print classifier

# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0)

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ')
