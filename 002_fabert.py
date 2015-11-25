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
dataset = 'fabert'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.580000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'weighting',
            'classifier:__choice__': 'extra_trees',
            'classifier:extra_trees:bootstrap': 'True',
            'classifier:extra_trees:criterion': 'gini',
            'classifier:extra_trees:max_depth': 'None',
            'classifier:extra_trees:max_features': 1.4927328322706173,
            'classifier:extra_trees:min_samples_leaf': 1,
            'classifier:extra_trees:min_samples_split': 5,
            'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
            'classifier:extra_trees:n_estimators': 100,
            'imputation:strategy': 'mean',
            'one_hot_encoding:use_minimum_fraction': 'False',
            'preprocessor:__choice__': 'select_rates',
            'preprocessor:select_rates:alpha': 0.4308279694614349,
            'preprocessor:select_rates:mode': 'fwe',
            'preprocessor:select_rates:score_func': 'f_classif',
            'rescaling:__choice__': 'min/max'})),
     (0.200000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'none',
            'classifier:__choice__': 'sgd',
            'classifier:sgd:alpha': 5.707045187542232e-06,
            'classifier:sgd:average': 'True',
            'classifier:sgd:eta0': 0.059208215107360226,
            'classifier:sgd:fit_intercept': 'True',
            'classifier:sgd:l1_ratio': 0.5696965689983325,
            'classifier:sgd:learning_rate': 'constant',
            'classifier:sgd:loss': 'log',
            'classifier:sgd:n_iter': 809,
            'classifier:sgd:penalty': 'elasticnet',
            'imputation:strategy': 'median',
            'one_hot_encoding:minimum_fraction': 0.45801169150718357,
            'one_hot_encoding:use_minimum_fraction': 'True',
            'preprocessor:__choice__': 'liblinear_svc_preprocessor',
            'preprocessor:liblinear_svc_preprocessor:C': 9.102297055334894,
            'preprocessor:liblinear_svc_preprocessor:dual': 'False',
            'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
            'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
            'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
            'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
            'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
            'preprocessor:liblinear_svc_preprocessor:tol': 9.129411357422978e-05,
            'rescaling:__choice__': 'normalize'})),
     (0.060000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'sgd',
             'classifier:sgd:alpha': 3.104241273548187e-05,
             'classifier:sgd:average': 'False',
             'classifier:sgd:eta0': 0.050396014246875294,
             'classifier:sgd:fit_intercept': 'True',
             'classifier:sgd:l1_ratio': 0.7121576951214108,
             'classifier:sgd:learning_rate': 'optimal',
             'classifier:sgd:loss': 'log',
             'classifier:sgd:n_iter': 649,
             'classifier:sgd:penalty': 'elasticnet',
             'imputation:strategy': 'mean',
             'one_hot_encoding:use_minimum_fraction': 'False',
             'preprocessor:__choice__': 'no_preprocessing',
             'rescaling:__choice__': 'min/max'})),
     (0.060000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'passive_aggressive',
             'classifier:passive_aggressive:C': 0.023003251414120036,
             'classifier:passive_aggressive:fit_intercept': 'True',
             'classifier:passive_aggressive:loss': 'hinge',
             'classifier:passive_aggressive:n_iter': 57,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.012167961375954476,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'liblinear_svc_preprocessor',
             'preprocessor:liblinear_svc_preprocessor:C': 0.07417606253933476,
             'preprocessor:liblinear_svc_preprocessor:dual': 'False',
             'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
             'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
             'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
             'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
             'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
             'preprocessor:liblinear_svc_preprocessor:tol': 0.0009557179607902859,
             'rescaling:__choice__': 'none'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'liblinear_svc',
             'classifier:liblinear_svc:C': 491.8319475226706,
             'classifier:liblinear_svc:dual': 'False',
             'classifier:liblinear_svc:fit_intercept': 'True',
             'classifier:liblinear_svc:intercept_scaling': 1,
             'classifier:liblinear_svc:loss': 'squared_hinge',
             'classifier:liblinear_svc:multi_class': 'ovr',
             'classifier:liblinear_svc:penalty': 'l2',
             'classifier:liblinear_svc:tol': 0.0008252238346618138,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.00028396835704950287,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'liblinear_svc_preprocessor',
             'preprocessor:liblinear_svc_preprocessor:C': 0.11029125786578071,
             'preprocessor:liblinear_svc_preprocessor:dual': 'False',
             'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
             'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
             'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
             'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
             'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
             'preprocessor:liblinear_svc_preprocessor:tol': 0.0003417183512181233,
             'rescaling:__choice__': 'min/max'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'sgd',
             'classifier:sgd:alpha': 2.618489922233997e-06,
             'classifier:sgd:average': 'False',
             'classifier:sgd:eta0': 0.0785971926323006,
             'classifier:sgd:fit_intercept': 'True',
             'classifier:sgd:l1_ratio': 0.1596938886542899,
             'classifier:sgd:learning_rate': 'constant',
             'classifier:sgd:loss': 'hinge',
             'classifier:sgd:n_iter': 509,
             'classifier:sgd:penalty': 'elasticnet',
             'imputation:strategy': 'mean',
             'one_hot_encoding:use_minimum_fraction': 'False',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.25578392394574817,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'chi2',
             'rescaling:__choice__': 'min/max'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'extra_trees',
             'classifier:extra_trees:bootstrap': 'False',
             'classifier:extra_trees:criterion': 'gini',
             'classifier:extra_trees:max_depth': 'None',
             'classifier:extra_trees:max_features': 2.1694048668692454,
             'classifier:extra_trees:min_samples_leaf': 1,
             'classifier:extra_trees:min_samples_split': 8,
             'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
             'classifier:extra_trees:n_estimators': 100,
             'imputation:strategy': 'median',
             'one_hot_encoding:minimum_fraction': 0.23760831456778012,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'no_preprocessing',
             'rescaling:__choice__': 'standardize'})), ]

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
