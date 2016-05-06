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
dataset = 'tania'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Use this version of lasagne commit of the lasagne master branch:
# 24c9ed2ffc25504c3b0df4598afb1e63fdd59eee
# https://github.com/Lasagne/Lasagne/commit/24c9ed2ffc25504c3b0df4598afb1e63fdd59eee
# Copy the file DeepFeedNet into autosklearn.pipeline.components.classification
# Copy the file FeedForwardNet into autosklearn.pipeline.implementations

choices = \
    [(0.220000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'none',
        'classifier:DeepFeedNet:activation': 'relu',
        'classifier:DeepFeedNet:batch_size': 1526,
        'classifier:DeepFeedNet:dropout_layer_1': 0.07375877191623954,
        'classifier:DeepFeedNet:dropout_layer_2': 0.25061726159515596,
        'classifier:DeepFeedNet:dropout_output': 0.44276742232825533,
        'classifier:DeepFeedNet:lambda2': 0.00559189810319557,
        'classifier:DeepFeedNet:learning_rate': 0.01,
        'classifier:DeepFeedNet:num_layers': 'd',
        'classifier:DeepFeedNet:num_units_layer_1': 3512,
        'classifier:DeepFeedNet:num_units_layer_2': 2456,
        'classifier:DeepFeedNet:number_updates': 1019,
        'classifier:DeepFeedNet:solver': 'smorm3s',
        'classifier:DeepFeedNet:std_layer_1': 0.0031572295374762784,
        'classifier:DeepFeedNet:std_layer_2': 0.024102151721155526,
        'classifier:__choice__': 'DeepFeedNet',
        'imputation:strategy': 'median',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:truncatedSVD:target_dim': 169,
        'preprocessor:__choice__': 'truncatedSVD',
        'rescaling:__choice__': 'normalize'})),
     (0.180000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'sgd',
         'classifier:sgd:alpha': 1e-06,
         'classifier:sgd:average': 'False',
         'classifier:sgd:eta0': 1e-07,
         'classifier:sgd:fit_intercept': 'True',
         'classifier:sgd:learning_rate': 'optimal',
         'classifier:sgd:loss': 'log',
         'classifier:sgd:n_iter': 5,
         'classifier:sgd:penalty': 'l2',
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     (0.140000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1526,
         'classifier:DeepFeedNet:dropout_layer_1': 0.07375877191623954,
         'classifier:DeepFeedNet:dropout_layer_2': 0.25061726159515596,
         'classifier:DeepFeedNet:dropout_output': 0.5318548466903714,
         'classifier:DeepFeedNet:lambda2': 0.00559189810319557,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'd',
         'classifier:DeepFeedNet:num_units_layer_1': 3512,
         'classifier:DeepFeedNet:num_units_layer_2': 2456,
         'classifier:DeepFeedNet:number_updates': 942,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0031572295374762784,
         'classifier:DeepFeedNet:std_layer_2': 0.024102151721155526,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 169,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.100000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1526,
         'classifier:DeepFeedNet:dropout_layer_1': 0.07375877191623954,
         'classifier:DeepFeedNet:dropout_layer_2': 0.25061726159515596,
         'classifier:DeepFeedNet:dropout_output': 0.5318548466903714,
         'classifier:DeepFeedNet:lambda2': 0.00559189810319557,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'd',
         'classifier:DeepFeedNet:num_units_layer_1': 2825,
         'classifier:DeepFeedNet:num_units_layer_2': 2456,
         'classifier:DeepFeedNet:number_updates': 942,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0031572295374762784,
         'classifier:DeepFeedNet:std_layer_2': 0.024102151721155526,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 169,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1526,
         'classifier:DeepFeedNet:dropout_layer_1': 0.07375877191623954,
         'classifier:DeepFeedNet:dropout_layer_2': 0.25061726159515596,
         'classifier:DeepFeedNet:dropout_output': 0.6315030660705527,
         'classifier:DeepFeedNet:lambda2': 0.00559189810319557,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'd',
         'classifier:DeepFeedNet:num_units_layer_1': 3512,
         'classifier:DeepFeedNet:num_units_layer_2': 2456,
         'classifier:DeepFeedNet:number_updates': 942,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0031572295374762784,
         'classifier:DeepFeedNet:std_layer_2': 0.024102151721155526,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 169,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.080000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 2124,
         'classifier:DeepFeedNet:dropout_layer_1': 0.01360549061849139,
         'classifier:DeepFeedNet:dropout_output': 0.2644391773986185,
         'classifier:DeepFeedNet:lambda2': 0.004871660362477711,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 2812,
         'classifier:DeepFeedNet:number_updates': 2710,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.09316319189582598,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 186,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1867,
         'classifier:DeepFeedNet:dropout_layer_1': 0.01908790794742743,
         'classifier:DeepFeedNet:dropout_output': 0.3448188758299382,
         'classifier:DeepFeedNet:lambda2': 0.0007755741149255707,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 3665,
         'classifier:DeepFeedNet:number_updates': 2512,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0024468150980905207,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.05266063283992454,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:truncatedSVD:target_dim': 166,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 2281,
         'classifier:DeepFeedNet:dropout_layer_1': 0.09094796094063819,
         'classifier:DeepFeedNet:dropout_output': 0.4958339054016198,
         'classifier:DeepFeedNet:lambda2': 1.805699319151882e-05,
         'classifier:DeepFeedNet:learning_rate': 0.001,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 2651,
         'classifier:DeepFeedNet:number_updates': 3403,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.007630682901621406,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'mean',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 197,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'none'})),
     (0.040000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 2086,
         'classifier:DeepFeedNet:dropout_layer_1': 0.1030823826758656,
         'classifier:DeepFeedNet:dropout_output': 0.22142344211272239,
         'classifier:DeepFeedNet:lambda2': 3.4109499881542005e-06,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 3317,
         'classifier:DeepFeedNet:number_updates': 711,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0012484056182083289,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.030925614928477674,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:truncatedSVD:target_dim': 159,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'min/max'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1336,
         'classifier:DeepFeedNet:dropout_layer_1': 0.0331786272132608,
         'classifier:DeepFeedNet:dropout_output': 0.3783990976694647,
         'classifier:DeepFeedNet:lambda2': 0.006318427713029419,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 2491,
         'classifier:DeepFeedNet:number_updates': 3437,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.09522419264016894,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:minimum_fraction': 0.03562984523180951,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:truncatedSVD:target_dim': 189,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1967,
         'classifier:DeepFeedNet:dropout_layer_1': 0.06971989322917795,
         'classifier:DeepFeedNet:dropout_output': 0.14345632673233852,
         'classifier:DeepFeedNet:lambda2': 0.0008778987660283575,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 3587,
         'classifier:DeepFeedNet:number_updates': 3182,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0015311970092555642,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 135,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 1882,
         'classifier:DeepFeedNet:dropout_layer_1': 0.007184660164183019,
         'classifier:DeepFeedNet:dropout_output': 0.35789769788034004,
         'classifier:DeepFeedNet:lambda2': 0.008162829194808478,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 3376,
         'classifier:DeepFeedNet:number_updates': 2868,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0010604662105437909,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:truncatedSVD:target_dim': 199,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:DeepFeedNet:activation': 'relu',
         'classifier:DeepFeedNet:batch_size': 2086,
         'classifier:DeepFeedNet:dropout_layer_1': 0.15565773821145037,
         'classifier:DeepFeedNet:dropout_output': 0.22142344211272239,
         'classifier:DeepFeedNet:lambda2': 1.7925329564209397e-06,
         'classifier:DeepFeedNet:learning_rate': 0.01,
         'classifier:DeepFeedNet:num_layers': 'c',
         'classifier:DeepFeedNet:num_units_layer_1': 3317,
         'classifier:DeepFeedNet:number_updates': 711,
         'classifier:DeepFeedNet:solver': 'smorm3s',
         'classifier:DeepFeedNet:std_layer_1': 0.0012484056182083289,
         'classifier:__choice__': 'DeepFeedNet',
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:minimum_fraction': 0.030925614928477674,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:truncatedSVD:target_dim': 159,
         'preprocessor:__choice__': 'truncatedSVD',
         'rescaling:__choice__': 'min/max'})),
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
