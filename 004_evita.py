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
dataset = 'evita'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.320000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'weighting',
        'classifier:__choice__': 'xgradient_boosting',
        'classifier:xgradient_boosting:base_score': 0.5,
        'classifier:xgradient_boosting:colsample_bylevel': 1,
        'classifier:xgradient_boosting:colsample_bytree': 1,
        'classifier:xgradient_boosting:gamma': 0,
        'classifier:xgradient_boosting:learning_rate': 0.083957576764175909,
        'classifier:xgradient_boosting:max_delta_step': 0,
        'classifier:xgradient_boosting:max_depth': 9,
        'classifier:xgradient_boosting:min_child_weight': 1,
        'classifier:xgradient_boosting:n_estimators': 207,
        'classifier:xgradient_boosting:reg_alpha': 0,
        'classifier:xgradient_boosting:reg_lambda': 1,
        'classifier:xgradient_boosting:scale_pos_weight': 1,
        'classifier:xgradient_boosting:subsample': 0.79041547139233681,
        'imputation:strategy': 'median',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:__choice__': 'select_rates',
        'preprocessor:select_rates:alpha': 0.033271689466917775,
        'preprocessor:select_rates:mode': 'fdr',
        'preprocessor:select_rates:score_func': 'chi2',
        'rescaling:__choice__': 'none'})),
     (0.140000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 1.0,
         'classifier:extra_trees:min_samples_leaf': 1,
         'classifier:extra_trees:min_samples_split': 2,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.10000000000000001,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.100000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 3.904721926856924,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 2,
         'classifier:random_forest:min_samples_split': 7,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.036176664478653142,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_percentile_classification',
         'preprocessor:select_percentile_classification:percentile': 91.78175624881186,
         'preprocessor:select_percentile_classification:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'True',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 1.0,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 1,
         'classifier:random_forest:min_samples_split': 2,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.18915206967606921,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'standardize'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 0.59875097583441961,
         'classifier:extra_trees:min_samples_leaf': 1,
         'classifier:extra_trees:min_samples_split': 2,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.13663946292601112,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'standardize'})),
     (0.060000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'True',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 1.0,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 1,
         'classifier:random_forest:min_samples_split': 2,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.10000000000000001,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'False',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 2.4071018354857294,
         'classifier:extra_trees:min_samples_leaf': 2,
         'classifier:extra_trees:min_samples_split': 9,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.34844304591109215,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 2.3037777871550227,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 1,
         'classifier:random_forest:min_samples_split': 6,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'standardize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'entropy',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 3.9417933307381925,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 2,
         'classifier:random_forest:min_samples_split': 3,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.076515481895064422,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.39998541946519961,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'True',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 2.6560184696178109,
         'classifier:extra_trees:min_samples_leaf': 1,
         'classifier:extra_trees:min_samples_split': 9,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.49576705570976692,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'extra_trees',
         'classifier:extra_trees:bootstrap': 'True',
         'classifier:extra_trees:criterion': 'gini',
         'classifier:extra_trees:max_depth': 'None',
         'classifier:extra_trees:max_features': 2.8762254807814838,
         'classifier:extra_trees:min_samples_leaf': 7,
         'classifier:extra_trees:min_samples_split': 7,
         'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
         'classifier:extra_trees:n_estimators': 100,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.00037525617209727315,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.36323622954313295,
         'preprocessor:select_rates:mode': 'fpr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'gini',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 4.7911724862642,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 1,
         'classifier:random_forest:min_samples_split': 11,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.47510655107871991,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'standardize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'entropy',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 4.9237570615905248,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 13,
         'classifier:random_forest:min_samples_split': 15,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.00028264986304734767,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'select_rates',
         'preprocessor:select_rates:alpha': 0.27910583898194102,
         'preprocessor:select_rates:mode': 'fdr',
         'preprocessor:select_rates:score_func': 'chi2',
         'rescaling:__choice__': 'none'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'random_forest',
         'classifier:random_forest:bootstrap': 'False',
         'classifier:random_forest:criterion': 'entropy',
         'classifier:random_forest:max_depth': 'None',
         'classifier:random_forest:max_features': 3.0988613659452917,
         'classifier:random_forest:max_leaf_nodes': 'None',
         'classifier:random_forest:min_samples_leaf': 3,
         'classifier:random_forest:min_samples_split': 3,
         'classifier:random_forest:min_weight_fraction_leaf': 0.0,
         'classifier:random_forest:n_estimators': 100,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
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
    predictions = predictions[:, 1].reshape((-1, 1))

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ', fmt='%.4e')
