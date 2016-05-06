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
dataset = 'helena'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.220000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'weighting',
        'classifier:__choice__': 'adaboost',
        'classifier:adaboost:algorithm': 'SAMME.R',
        'classifier:adaboost:learning_rate': 0.12736378214916136,
        'classifier:adaboost:max_depth': 2,
        'classifier:adaboost:n_estimators': 102,
        'imputation:strategy': 'mean',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:__choice__': 'no_preprocessing',
        'rescaling:__choice__': 'min/max'})),
     (0.140000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 34.52330718740001,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.010305332230700001,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.00012464201046600006,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'polynomial',
         'preprocessor:polynomial:degree': 2,
         'preprocessor:polynomial:include_bias': 'True',
         'preprocessor:polynomial:interaction_only': 'False',
         'rescaling:__choice__': 'none'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'True',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 1.1473936812138448,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 11,
         'classifier:random_forest:min_samples_split': 10,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'fast_ica',
         'preprocessor:fast_ica:algorithm': 'parallel',
         'preprocessor:fast_ica:fun': 'logcosh',
         'preprocessor:fast_ica:n_components': 945,
         'preprocessor:fast_ica:whiten': 'True',
         'rescaling:__choice__': 'none'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 1.3455409527727558,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'pca',
         'preprocessor:pca:keep_variance': 0.7598172817638718,
         'preprocessor:pca:whiten': 'False',
         'rescaling:__choice__': 'standardize'})),
     (0.060000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 7.873556221817867,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.007384474684230516,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.44352666713957484,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'f_classif',
         'rescaling:__choice__': 'normalize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'lda',
         'classifier:lda:n_components': 12,
         'classifier:lda:shrinkage': 'manual',
         'classifier:lda:shrinkage_factor': 0.9016175646665451,
         'classifier:lda:tol': 0.0001716207118446579,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.009728842857612658,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'fast_ica',
         'preprocessor:fast_ica:algorithm': 'parallel',
         'preprocessor:fast_ica:fun': 'logcosh',
         'preprocessor:fast_ica:n_components': 914,
         'preprocessor:fast_ica:whiten': 'True',
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 4.213462678722325,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.020501216047798837,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'f_classif',
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 4.367371232039595,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.01303718715506049,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 7.286051530772571,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.026747542179073727,
         'preprocessor:select_rates:mode': 'fwe',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'min/max'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 7.907981363846062,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.38925641117203025,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'f_classif',
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 7.873556221817867,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.007384474684230516,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 709.0694499917347,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 0.013228763477510586,
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 3.0174234734498917,
         'classifier:extra_trees:min_samples_leaf': 2,
         'classifier:extra_trees:min_samples_split': 12,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.007553789957243724,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 31.20787569423215,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 1.7149340429765088e-05,
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 4.592027482980136,
         'classifier:extra_trees:min_samples_leaf': 12,
         'classifier:extra_trees:min_samples_split': 12,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.003355962206220629,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 0.14162959993684351,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 0.009394425053603682,
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 2.655307919311661,
         'classifier:extra_trees:min_samples_leaf': 2,
         'classifier:extra_trees:min_samples_split': 16,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.00019806605573813597,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 18.30206355212093,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 3.267407083806816e-05,
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 1.2325644317889806,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.01303718715506049,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 7.286051530772571,
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'extra_trees_preproc_for_classification',
         'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'False',
         'preprocessor:extra_trees_preproc_for_classification:criterion': 'entropy',
         'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None',
         'preprocessor:extra_trees_preproc_for_classification:max_features': 1.3440864854665975,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 8,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 17,
         'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0,
         'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100,
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 8.519756045823158,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.08901572125739037,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'extra_trees_preproc_for_classification',
         'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'False',
         'preprocessor:extra_trees_preproc_for_classification:criterion': 'entropy',
         'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None',
         'preprocessor:extra_trees_preproc_for_classification:max_features': 3.842249530515841,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 13,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 10,
         'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0,
         'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100,
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 3.7289920990557777,
         'classifier:extra_trees:min_samples_leaf': 3,
         'classifier:extra_trees:min_samples_split': 13,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.00037734441447340595,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 0.6186775496832956,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 1.710156140413348e-05,
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 4.430190032276566,
         'classifier:extra_trees:min_samples_leaf': 5,
         'classifier:extra_trees:min_samples_split': 9,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.0027303638882864483,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 30343.867455246524,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 0.005743178077382402,
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 1.8440666453536427,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 1,
         'classifier:random_forest:min_samples_split': 14,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'liblinear_svc_preprocessor',
         'preprocessor:liblinear_svc_preprocessor:C': 28279.093774727116,
         'preprocessor:liblinear_svc_preprocessor:dual': 'False',
         'preprocessor:liblinear_svc_preprocessor:fit_intercept': 'True',
         'preprocessor:liblinear_svc_preprocessor:intercept_scaling': 1,
         'preprocessor:liblinear_svc_preprocessor:loss': 'squared_hinge',
         'preprocessor:liblinear_svc_preprocessor:multi_class': 'ovr',
         'preprocessor:liblinear_svc_preprocessor:penalty': 'l1',
         'preprocessor:liblinear_svc_preprocessor:tol': 0.0010803540483296555,
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'qda',
         'classifier:qda:reg_param': 1.3455409527727558,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.04805977625874754,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'extra_trees_preproc_for_classification',
         'preprocessor:extra_trees_preproc_for_classification:bootstrap': 'False',
         'preprocessor:extra_trees_preproc_for_classification:criterion': 'entropy',
         'preprocessor:extra_trees_preproc_for_classification:max_depth': 'None',
         'preprocessor:extra_trees_preproc_for_classification:max_features': 3.6600607240096594,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_leaf': 18,
         'preprocessor:extra_trees_preproc_for_classification:min_samples_split': 18,
         'preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf': 0.0,
         'preprocessor:extra_trees_preproc_for_classification:n_estimators': 100,
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
