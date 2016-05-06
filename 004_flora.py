import argparse
import os

from joblib import Parallel, delayed
import numpy as np

import autosklearn
import autosklearn.data
import autosklearn.data.competition_data_manager
from autosklearn.pipeline.regression import SimpleRegressionPipeline

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

input = args.input
dataset = 'flora'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.220000, SimpleRegressionPipeline(configuration={
        'imputation:strategy': 'most_frequent',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:__choice__': 'no_preprocessing',
        'regressor:__choice__': 'xgradient_boosting',
        'regressor:xgradient_boosting:base_score': 0.5,
        'regressor:xgradient_boosting:colsample_bylevel': 1,
        'regressor:xgradient_boosting:colsample_bytree': 1,
        'regressor:xgradient_boosting:gamma': 0,
        'regressor:xgradient_boosting:learning_rate': 0.056838908807173093,
        'regressor:xgradient_boosting:max_delta_step': 0,
        'regressor:xgradient_boosting:max_depth': 8,
        'regressor:xgradient_boosting:min_child_weight': 16,
        'regressor:xgradient_boosting:n_estimators': 178,
        'regressor:xgradient_boosting:reg_alpha': 0,
        'regressor:xgradient_boosting:reg_lambda': 1,
        'regressor:xgradient_boosting:scale_pos_weight': 1,
        'regressor:xgradient_boosting:subsample': 0.70026686345272005,
        'rescaling:__choice__': 'none'})),
     (0.160000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.028721299365033225,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 6,
         'regressor:xgradient_boosting:min_child_weight': 13,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.120000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.00076890296310299397,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 8,
         'regressor:xgradient_boosting:min_child_weight': 1,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.080000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 7,
         'regressor:xgradient_boosting:min_child_weight': 1,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.080000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.0023636879664826662,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'liblinear_svr',
         'regressor:liblinear_svr:C': 1756.3281019761341,
         'regressor:liblinear_svr:dual': 'False',
         'regressor:liblinear_svr:epsilon': 0.12958135960591446,
         'regressor:liblinear_svr:fit_intercept': 'True',
         'regressor:liblinear_svr:intercept_scaling': 1,
         'regressor:liblinear_svr:loss': 'squared_epsilon_insensitive',
         'regressor:liblinear_svr:tol': 6.7973376271281637e-05,
         'rescaling:__choice__': 'none'})),
     (0.060000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.0078832566242014457,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'kernel_pca',
         'preprocessor:kernel_pca:coef0': 0.830468268944067,
         'preprocessor:kernel_pca:kernel': 'sigmoid',
         'preprocessor:kernel_pca:n_components': 1297,
         'regressor:__choice__': 'sgd',
         'regressor:sgd:alpha': 7.1922597888891864e-06,
         'regressor:sgd:average': 'True',
         'regressor:sgd:epsilon': 0.002325854486140731,
         'regressor:sgd:eta0': 0.09745049410405518,
         'regressor:sgd:fit_intercept': 'True',
         'regressor:sgd:learning_rate': 'invscaling',
         'regressor:sgd:loss': 'squared_epsilon_insensitive',
         'regressor:sgd:n_iter': 56,
         'regressor:sgd:penalty': 'l1',
         'regressor:sgd:power_t': 0.2820868931235419,
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.39354372832974382,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 3,
         'regressor:xgradient_boosting:min_child_weight': 19,
         'regressor:xgradient_boosting:n_estimators': 73,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 0.51160818820515941,
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.0001292396238727452,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 5,
         'regressor:xgradient_boosting:min_child_weight': 1,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.040000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.0010042712846593592,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'extra_trees_preproc_for_regression',
         'preprocessor:extra_trees_preproc_for_regression:bootstrap': 'False',
         'preprocessor:extra_trees_preproc_for_regression:criterion': 'mse',
         'preprocessor:extra_trees_preproc_for_regression:max_depth': 'None',
         'preprocessor:extra_trees_preproc_for_regression:max_features': 4.4366238138449141,
         'preprocessor:extra_trees_preproc_for_regression:min_samples_leaf': 5,
         'preprocessor:extra_trees_preproc_for_regression:min_samples_split': 2,
         'preprocessor:extra_trees_preproc_for_regression:min_weight_fraction_leaf': 0.0,
         'preprocessor:extra_trees_preproc_for_regression:n_estimators': 100,
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.24786184996967336,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 4,
         'regressor:xgradient_boosting:min_child_weight': 12,
         'regressor:xgradient_boosting:n_estimators': 487,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 0.51768561001523961,
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.056838908807173093,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 6,
         'regressor:xgradient_boosting:min_child_weight': 20,
         'regressor:xgradient_boosting:n_estimators': 178,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 0.81655152788480145,
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'truncatedSVD',
         'preprocessor:truncatedSVD:target_dim': 222,
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 3,
         'regressor:xgradient_boosting:min_child_weight': 1,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'truncatedSVD',
         'preprocessor:truncatedSVD:target_dim': 156,
         'regressor:__choice__': 'decision_tree',
         'regressor:decision_tree:criterion': 'mse',
         'regressor:decision_tree:max_depth': 1.4573346058635357,
         'regressor:decision_tree:max_features': 1.0,
         'regressor:decision_tree:max_leaf_nodes': 'None',
         'regressor:decision_tree:min_samples_leaf': 17,
         'regressor:decision_tree:min_samples_split': 8,
         'regressor:decision_tree:min_weight_fraction_leaf': 0.0,
         'regressor:decision_tree:splitter': 'best',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'xgradient_boosting',
         'regressor:xgradient_boosting:base_score': 0.5,
         'regressor:xgradient_boosting:colsample_bylevel': 1,
         'regressor:xgradient_boosting:colsample_bytree': 1,
         'regressor:xgradient_boosting:gamma': 0,
         'regressor:xgradient_boosting:learning_rate': 0.10000000000000002,
         'regressor:xgradient_boosting:max_delta_step': 0,
         'regressor:xgradient_boosting:max_depth': 5,
         'regressor:xgradient_boosting:min_child_weight': 13,
         'regressor:xgradient_boosting:n_estimators': 100,
         'regressor:xgradient_boosting:reg_alpha': 0,
         'regressor:xgradient_boosting:reg_lambda': 1,
         'regressor:xgradient_boosting:scale_pos_weight': 1,
         'regressor:xgradient_boosting:subsample': 1.0,
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.0030893906804030156,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'truncatedSVD',
         'preprocessor:truncatedSVD:target_dim': 67,
         'regressor:__choice__': 'k_nearest_neighbors',
         'regressor:k_nearest_neighbors:n_neighbors': 29,
         'regressor:k_nearest_neighbors:p': 2,
         'regressor:k_nearest_neighbors:weights': 'distance',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.0027171559129851464,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'truncatedSVD',
         'preprocessor:truncatedSVD:target_dim': 35,
         'regressor:__choice__': 'liblinear_svr',
         'regressor:liblinear_svr:C': 0.0485964760119761,
         'regressor:liblinear_svr:dual': 'False',
         'regressor:liblinear_svr:epsilon': 0.01333919934708307,
         'regressor:liblinear_svr:fit_intercept': 'True',
         'regressor:liblinear_svr:intercept_scaling': 1,
         'regressor:liblinear_svr:loss': 'squared_epsilon_insensitive',
         'regressor:liblinear_svr:tol': 0.030573671793931671,
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:__choice__': 'decision_tree',
         'regressor:decision_tree:criterion': 'mse',
         'regressor:decision_tree:max_depth': 0.031442410091469419,
         'regressor:decision_tree:max_features': 1.0,
         'regressor:decision_tree:max_leaf_nodes': 'None',
         'regressor:decision_tree:min_samples_leaf': 15,
         'regressor:decision_tree:min_samples_split': 10,
         'regressor:decision_tree:min_weight_fraction_leaf': 0.0,
         'regressor:decision_tree:splitter': 'best',
         'rescaling:__choice__': 'normalize'})),
     ]

targets = []
predictions = []
predictions_valid = []
predictions_test = []


def fit_and_predict(estimator, weight, X, y):
    try:
        estimator.fit(X.copy(), y.copy())
        pv = estimator.predict(X_valid.copy()) * weight
        pt = estimator.predict(X_test.copy()) * weight
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
