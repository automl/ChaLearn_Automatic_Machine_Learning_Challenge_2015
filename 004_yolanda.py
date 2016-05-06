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
dataset = 'yolanda'
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
# Copy the file RegDeepNet into autosklearn.pipeline.components.regression
# Copy the file FeedForwardNet into autosklearn.pipeline.implementations

choices = \
    [(0.360000, SimpleRegressionPipeline(configuration={
        'imputation:strategy': 'mean',
        'one_hot_encoding:minimum_fraction': 0.049682918006307676,
        'one_hot_encoding:use_minimum_fraction': 'True',
        'preprocessor:__choice__': 'no_preprocessing',
        'regressor:RegDeepNet:activation': 'tanh',
        'regressor:RegDeepNet:batch_size': 1865,
        'regressor:RegDeepNet:dropout_layer_1': 0.017462492577406473,
        'regressor:RegDeepNet:dropout_layer_2': 0.048354205627225436,
        'regressor:RegDeepNet:dropout_output': 0.00962149073006804,
        'regressor:RegDeepNet:lambda2': 1.0282444549550921e-05,
        'regressor:RegDeepNet:learning_rate': 0.001,
        'regressor:RegDeepNet:num_layers': 'd',
        'regressor:RegDeepNet:num_units_layer_1': 2615,
        'regressor:RegDeepNet:num_units_layer_2': 252,
        'regressor:RegDeepNet:number_updates': 3225,
        'regressor:RegDeepNet:solver': 'smorm3s',
        'regressor:RegDeepNet:std_layer_1': 0.006861129306844183,
        'regressor:RegDeepNet:std_layer_2': 0.002395977520245193,
        'regressor:__choice__': 'RegDeepNet',
        'rescaling:__choice__': 'standardize'})),
     (0.320000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.05112532429613385,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:RegDeepNet:activation': 'sigmoid',
         'regressor:RegDeepNet:batch_size': 1840,
         'regressor:RegDeepNet:dropout_layer_1': 0.15186663743978646,
         'regressor:RegDeepNet:dropout_layer_2': 0.11387781420379316,
         'regressor:RegDeepNet:dropout_layer_3': 0.19220971946536616,
         'regressor:RegDeepNet:dropout_output': 0.5509953660515314,
         'regressor:RegDeepNet:lambda2': 2.3655442216865217e-06,
         'regressor:RegDeepNet:learning_rate': 0.1,
         'regressor:RegDeepNet:num_layers': 'e',
         'regressor:RegDeepNet:num_units_layer_1': 173,
         'regressor:RegDeepNet:num_units_layer_2': 690,
         'regressor:RegDeepNet:num_units_layer_3': 2761,
         'regressor:RegDeepNet:number_updates': 4173,
         'regressor:RegDeepNet:solver': 'smorm3s',
         'regressor:RegDeepNet:std_layer_1': 0.006483588902887654,
         'regressor:RegDeepNet:std_layer_2': 0.006696161430555593,
         'regressor:RegDeepNet:std_layer_3': 0.0030798462419321746,
         'regressor:__choice__': 'RegDeepNet',
         'rescaling:__choice__': 'standardize'})),
     (0.160000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.00044746581915706805,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:RegDeepNet:activation': 'tanh',
         'regressor:RegDeepNet:batch_size': 1867,
         'regressor:RegDeepNet:dropout_layer_1': 0.0044842379741719856,
         'regressor:RegDeepNet:dropout_output': 0.029970881815609602,
         'regressor:RegDeepNet:lambda2': 3.922344043854585e-05,
         'regressor:RegDeepNet:learning_rate': 0.001,
         'regressor:RegDeepNet:num_layers': 'c',
         'regressor:RegDeepNet:num_units_layer_1': 2775,
         'regressor:RegDeepNet:number_updates': 4672,
         'regressor:RegDeepNet:solver': 'smorm3s',
         'regressor:RegDeepNet:std_layer_1': 0.0011091871005401157,
         'regressor:__choice__': 'RegDeepNet',
         'rescaling:__choice__': 'standardize'})),
     (0.100000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.0006151267694526832,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:RegDeepNet:activation': 'tanh',
         'regressor:RegDeepNet:batch_size': 1293,
         'regressor:RegDeepNet:dropout_layer_1': 0.024322298790122678,
         'regressor:RegDeepNet:dropout_layer_2': 0.4831886801640319,
         'regressor:RegDeepNet:dropout_layer_3': 0.7303058944461246,
         'regressor:RegDeepNet:dropout_output': 0.43112081941910074,
         'regressor:RegDeepNet:lambda2': 4.561723820100022e-06,
         'regressor:RegDeepNet:learning_rate': 0.001,
         'regressor:RegDeepNet:num_layers': 'e',
         'regressor:RegDeepNet:num_units_layer_1': 2999,
         'regressor:RegDeepNet:num_units_layer_2': 1630,
         'regressor:RegDeepNet:num_units_layer_3': 897,
         'regressor:RegDeepNet:number_updates': 4471,
         'regressor:RegDeepNet:solver': 'smorm3s',
         'regressor:RegDeepNet:std_layer_1': 0.0013646791717249367,
         'regressor:RegDeepNet:std_layer_2': 0.012431732856634247,
         'regressor:RegDeepNet:std_layer_3': 0.002351992156794049,
         'regressor:__choice__': 'RegDeepNet',
         'rescaling:__choice__': 'standardize'})),
     (0.060000, SimpleRegressionPipeline(configuration={
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.006283026157824821,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'regressor:RegDeepNet:activation': 'tanh',
         'regressor:RegDeepNet:batch_size': 1802,
         'regressor:RegDeepNet:dropout_layer_1': 0.01257793094940521,
         'regressor:RegDeepNet:dropout_output': 0.023821950297696383,
         'regressor:RegDeepNet:lambda2': 8.078248563082777e-05,
         'regressor:RegDeepNet:learning_rate': 0.001,
         'regressor:RegDeepNet:num_layers': 'c',
         'regressor:RegDeepNet:num_units_layer_1': 3293,
         'regressor:RegDeepNet:number_updates': 4842,
         'regressor:RegDeepNet:solver': 'smorm3s',
         'regressor:RegDeepNet:std_layer_1': 0.001130906938022124,
         'regressor:__choice__': 'RegDeepNet',
         'rescaling:__choice__': 'standardize'})),
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
