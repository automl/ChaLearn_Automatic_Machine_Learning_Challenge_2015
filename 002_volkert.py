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
dataset = 'volkert'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.480000, ParamSklearnClassifier(configuration={
        'balancing:strategy': 'none',
        'classifier:__choice__': 'random_forest',
        'classifier:random_forest:bootstrap': 'True',
        'classifier:random_forest:criterion': 'entropy',
        'classifier:random_forest:max_depth': 'None',
        'classifier:random_forest:max_features': 4.885151102990943,
        'classifier:random_forest:max_leaf_nodes': 'None',
        'classifier:random_forest:min_samples_leaf': 2,
        'classifier:random_forest:min_samples_split': 2,
        'classifier:random_forest:min_weight_fraction_leaf': 0.0,
        'classifier:random_forest:n_estimators': 100,
        'imputation:strategy': 'median',
        'one_hot_encoding:minimum_fraction': 0.059297498551361,
        'one_hot_encoding:use_minimum_fraction': 'True',
        'preprocessor:__choice__': 'gem',
        'preprocessor:gem:N': 13,
        'preprocessor:gem:precond': 0.31299029323203487,
        'rescaling:__choice__': 'min/max'})),
     (0.300000, ParamSklearnClassifier(
        configuration={
            'balancing:strategy': 'none',
            'classifier:__choice__': 'random_forest',
            'classifier:random_forest:bootstrap': 'False',
            'classifier:random_forest:criterion': 'entropy',
            'classifier:random_forest:max_depth': 'None',
            'classifier:random_forest:max_features': 4.908992016092793,
            'classifier:random_forest:max_leaf_nodes': 'None',
            'classifier:random_forest:min_samples_leaf': 2,
            'classifier:random_forest:min_samples_split': 6,
            'classifier:random_forest:min_weight_fraction_leaf': 0.0,
            'classifier:random_forest:n_estimators': 100,
            'imputation:strategy': 'mean',
            'one_hot_encoding:minimum_fraction': 0.009349768412523697,
            'one_hot_encoding:use_minimum_fraction': 'True',
            'preprocessor:__choice__': 'fast_ica',
            'preprocessor:fast_ica:algorithm': 'deflation',
            'preprocessor:fast_ica:fun': 'exp',
            'preprocessor:fast_ica:whiten': 'False',
            'rescaling:__choice__': 'none'})),
     (0.180000,
        ParamSklearnClassifier(
            configuration={
                'balancing:strategy': 'weighting',
                'classifier:__choice__': 'libsvm_svc',
                'classifier:libsvm_svc:C': 445.91825904609124,
                'classifier:libsvm_svc:gamma': 0.03873498413280048,
                'classifier:libsvm_svc:kernel': 'rbf',
                'classifier:libsvm_svc:max_iter': -1,
                'classifier:libsvm_svc:shrinking': 'True',
                'classifier:libsvm_svc:tol': 0.0008078719040695308,
                'imputation:strategy': 'median',
                'one_hot_encoding:use_minimum_fraction': 'False',
                'preprocessor:__choice__': 'pca',
                'preprocessor:pca:keep_variance': 0.7596970304901425,
                'preprocessor:pca:whiten': 'True',
                'rescaling:__choice__': 'standardize'})),
     (0.040000, ParamSklearnClassifier(
         configuration={
             'balancing:strategy': 'none',
             'classifier:__choice__': 'random_forest',
             'classifier:random_forest:bootstrap': 'False',
             'classifier:random_forest:criterion': 'entropy',
             'classifier:random_forest:max_depth': 'None',
             'classifier:random_forest:max_features': 3.5340547102377364,
             'classifier:random_forest:max_leaf_nodes': 'None',
             'classifier:random_forest:min_samples_leaf': 2,
             'classifier:random_forest:min_samples_split': 6,
             'classifier:random_forest:min_weight_fraction_leaf': 0.0,
             'classifier:random_forest:n_estimators': 100,
             'imputation:strategy': 'mean',
             'one_hot_encoding:minimum_fraction': 0.008518947433195237,
             'one_hot_encoding:use_minimum_fraction': 'True',
             'preprocessor:__choice__': 'fast_ica',
             'preprocessor:fast_ica:algorithm': 'deflation',
             'preprocessor:fast_ica:fun': 'cube',
             'preprocessor:fast_ica:whiten': 'False',
             'rescaling:__choice__': 'none'})), ]

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
