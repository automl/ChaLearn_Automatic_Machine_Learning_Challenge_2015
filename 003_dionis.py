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
dataset = 'dionis'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.520000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'none',
        'classifier:__choice__': 'qda',
        'classifier:qda:reg_param': 7.017044041208607,
        'imputation:strategy': 'most_frequent',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:__choice__': 'no_preprocessing',
        'rescaling:__choice__': 'normalize'})),
     (0.360000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 0.5,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.1,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'k_nearest_neighbors',
         'classifier:k_nearest_neighbors:n_neighbors': 53,
         'classifier:k_nearest_neighbors:p': 2,
         'classifier:k_nearest_neighbors:weights': 'uniform',
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.004107223932117523,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.06365705922416094,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'f_classif',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 1288.9425457179896,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 6.852190351970404e-05,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.016322736180045382,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.48582026589548283,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 0.6872563090086077,
         'classifier:extra_trees:min_samples_leaf': 9,
         'classifier:extra_trees:min_samples_split': 8,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.00048281479349728755,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'feature_agglomeration',
         'preprocessor:feature_agglomeration:affinity': 'manhattan',
         'preprocessor:feature_agglomeration:linkage': 'average',
         'preprocessor:feature_agglomeration:n_clusters': 170,
         'preprocessor:feature_agglomeration:pooling_func': 'mean',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 737.3354222113379,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.029993063054990464,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.0007084092083452885,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.28020088992913833,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'f_classif',
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'k_nearest_neighbors',
         'classifier:k_nearest_neighbors:n_neighbors': 1,
         'classifier:k_nearest_neighbors:p': 2,
         'classifier:k_nearest_neighbors:weights': 'uniform',
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.015690633649222446,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 1.0,
         'classifier:extra_trees:min_samples_leaf': 10,
         'classifier:extra_trees:min_samples_split': 2,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.01,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.1,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
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
