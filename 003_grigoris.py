import argparse
import os

from joblib import Parallel, delayed
import numpy as np

import autosklearn
import autosklearn.data
import autosklearn.data.competition_data_manager
from autosklearn.pipeline.classification import SimpleClassificationPipeline

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

input = args.input
dataset = 'grigoris'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.720000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'none',
        'classifier:__choice__': 'liblinear_svc',
        'classifier:liblinear_svc:C': 0.0665747065156058,
        'classifier:liblinear_svc:dual': 'False',
        'classifier:liblinear_svc:fit_intercept': 'True',
        'classifier:liblinear_svc:intercept_scaling': 1,
        'classifier:liblinear_svc:loss': 'squared_hinge',
        'classifier:liblinear_svc:multi_class': 'ovr',
        'classifier:liblinear_svc:penalty': 'l2',
        'classifier:liblinear_svc:tol': 0.002362381246384099,
        'imputation:strategy': 'mean',
        'one_hot_encoding:minimum_fraction': 0.0972585384393519,
        'one_hot_encoding:use_minimum_fraction': 'True',
        'preprocessor:__choice__': 'no_preprocessing',
        'rescaling:__choice__': 'normalize'})),
     (0.100000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 7.705276414124367,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.028951969755081776,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 1.0,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.0001,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.0033856971814438443,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 0.2598769185905466,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.001007160236770467,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.019059927375795167,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 0.6849477125990308,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 1.2676147487949745e-05,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.003803817610653382,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     ]

targets = []
predictions = []
predictions_valid = []
predictions_test = []


def fit_and_predict(estimator, weight, X, y):
    try:
        estimator.fit(X.copy(), y.copy())
        pv = estimator.predict_proba(X_valid.copy()) * weight
        pt = estimator.predict_proba(X_test.copy()) * weight
    except Exception as e:
        print(e)
        print(estimator.configuration)
        pv = None
        pt = None
    return pv, pt


# Make predictions and weight them
all_predictions = Parallel(n_jobs=-1)(delayed(fit_and_predict) \
                                          (estimator, weight, X, y) for
                                      weight, estimator in choices)
for pv, pt in all_predictions:
    predictions_valid.append(pv)
    predictions_test.append(pt)

# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0).astype(np.float32)

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ', fmt='%.4e')
