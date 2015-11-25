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
dataset = 'robert'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.140000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'none',
            'classifier:__choice__': 'random_forest',
            'classifier:random_forest:bootstrap': 'False',
            'classifier:random_forest:criterion': 'gini',
            'classifier:random_forest:max_depth': 'None',
            'classifier:random_forest:max_features': 4.649151092701434,
            'classifier:random_forest:max_leaf_nodes': 'None',
            'classifier:random_forest:min_samples_leaf': 3,
            'classifier:random_forest:min_samples_split': 5,
            'classifier:random_forest:min_weight_fraction_leaf': 0.0,
            'classifier:random_forest:n_estimators': 100,
            'imputation:strategy': 'most_frequent',
            'one_hot_encoding:minimum_fraction': 0.006861808529548735,
            'one_hot_encoding:use_minimum_fraction': 'True',
            'preprocessor:__choice__': 'select_rates',
            'preprocessor:select_rates:alpha': 0.03408255008474342,
            'preprocessor:select_rates:mode': 'fwe',
            'preprocessor:select_rates:score_func': 'f_classif',
            'rescaling:__choice__': 'normalize'})),
     (0.100000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.903953547277064,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 7,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:use_minimum_fraction': 'False',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.39048955069635144,
             'preprocessor:select_rates:mode': 'fdr',
             'preprocessor:select_rates:score_func': 'chi2',
             'rescaling:__choice__': 'min/max'})),
     (0.100000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.993159755057918,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 7,
             'classifier:random_forest:min_samples_split': 8,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.012773692146378994,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.4253112422037886,
             'preprocessor:select_rates:mode': 'fwe',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.080000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.824231595074576,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 2,
             'classifier:random_forest:min_samples_split': 9,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.015409179231947682,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.15855293870881282,
             'preprocessor:select_rates:mode': 'fwe',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.080000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.986249630554656,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 9,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.0004094368348333537,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.2042184046992981,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.080000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.979543436460854,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 13,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.0030276030286703853,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.2553320370022284,
             'preprocessor:select_rates:mode': 'fdr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.080000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'libsvm_svc',
             'classifier:libsvm_svc:C': 32410.92354891328,
             'classifier:libsvm_svc:gamma': 0.00032031918207606175,
             'classifier:libsvm_svc:kernel': 'rbf',
             'classifier:libsvm_svc:max_iter': -1,
             'classifier:libsvm_svc:shrinking': 'False',
             'classifier:libsvm_svc:tol': 0.00010267502073438567,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.2563807302883408,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'extra_trees_preproc_for_classification',
             'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'False',
             'preprocessor:extra_trees_preproc_for_classification:criterion': 'entropy',
             'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None',
             'preprocessor:extra_trees_preproc_for_classification:max_features': 1.0992480125604416,
             'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 14,
             'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 14,
             'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0,
             'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100,
             'rescaling:__choice__': 'standardize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.9272757303139345,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 6,
             'classifier:random_forest:min_samples_split': 11,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.026585945335069572,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.18365680124572498,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.317500812299693,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 4,
             'classifier:random_forest:min_samples_split': 10,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.07211758028194257,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.413921654101964,
             'preprocessor:select_rates:mode': 'fwe',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.979543436460854,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 11,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.3457748242188964,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.2553320370022284,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.984094209843432,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 2,
             'classifier:random_forest:min_samples_split': 6,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.002964965965195125,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.32902835724886753,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.976498799860011,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 11,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.00023264482985881722,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.18599313694852976,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.8771232490057645,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 5,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.0036299052612188843,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'extra_trees_preproc_for_classification',
             'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'False',
             'preprocessor:extra_trees_preproc_for_classification:criterion': 'gini',
             'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None',
             'preprocessor:extra_trees_preproc_for_classification:max_features': 3.227481830799948,
             'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 10,
             'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 15,
             'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0,
             'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100,
             'rescaling:__choice__': 'none'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'weighting',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.9972300922461965,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 1,
             'classifier:random_forest:min_samples_split': 5,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.0036710365644770974,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.2824253935610728,
             'preprocessor:select_rates:mode': 'fwe',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'min/max'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.987558297214975,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 7,
             'classifier:random_forest:min_samples_split': 12,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.009762178652802867,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.15249382748406307,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.916387010790206,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 6,
             'classifier:random_forest:min_samples_split': 6,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.004208471063037365,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.31136291562274643,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.742010306437378,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 5,
             'classifier:random_forest:min_samples_split': 7,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.07864330044363266,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.32710038856858903,
             'preprocessor:select_rates:mode': 'fpr',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})),
     (0.020000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'gini',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 4.261790364868923,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 3,
             'classifier:random_forest:min_samples_split': 10,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'most_frequent',
             'one_hot_encoding:minimum_fraction': 0.10391104479398042,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'select_rates',
             'preprocessor:select_rates:alpha': 0.1,
             'preprocessor:select_rates:mode': 'fwe',
             'preprocessor:select_rates:score_func': 'f_classif',
             'rescaling:__choice__': 'normalize'})), ]

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
