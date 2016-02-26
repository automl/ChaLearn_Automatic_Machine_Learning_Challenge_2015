import argparse
import os

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
dataset = 'wallis'
output = args.output

path = os.path.join(input, dataset)
D = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Replace the following array by a new ensemble
choices = \
    [(0.580000, SimpleClassificationPipeline(configuration={
        'balancing:strategy': 'weighting',
        'classifier:__choice__': 'passive_aggressive',
        'classifier:passive_aggressive:C': 0.0006373873391108438,
        'classifier:passive_aggressive:fit_intercept': 'True',
        'classifier:passive_aggressive:loss': 'squared_hinge',
        'classifier:passive_aggressive:n_iter': 18,
        'imputation:strategy': 'median',
        'one_hot_encoding:use_minimum_fraction': 'False',
        'preprocessor:__choice__': 'no_preprocessing',
        'rescaling:__choice__': 'normalize'})),
     (0.200000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'weighting',
         'classifier:__choice__': 'passive_aggressive',
         'classifier:passive_aggressive:C': 0.000465329983806252,
         'classifier:passive_aggressive:fit_intercept': 'True',
         'classifier:passive_aggressive:loss': 'squared_hinge',
         'classifier:passive_aggressive:n_iter': 34,
         'imputation:strategy': 'median',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'kernel_pca',
         'preprocessor:kernel_pca:kernel': 'cosine',
         'preprocessor:kernel_pca:n_components': 1351,
         'rescaling:__choice__': 'normalize'})),
     (0.180000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 0.7416809477859192,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.0048882934000166346,
         'imputation:strategy': 'most_frequent',
         'one_hot_encoding:use_minimum_fraction': 'False',
         'preprocessor:__choice__': 'select_percentile_classification',
         'preprocessor:select_percentile_classification:percentile': 19.775149789978155,
         'preprocessor:select_percentile_classification:score_func': 'chi2',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 0.4010081266689033,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.003197120920655818,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.0002497904559463802,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'no_preprocessing',
         'rescaling:__choice__': 'normalize'})),
     (0.020000, SimpleClassificationPipeline(configuration={
         'balancing:strategy': 'none',
         'classifier:__choice__': 'liblinear_svc',
         'classifier:liblinear_svc:C': 0.7444178979935873,
         'classifier:liblinear_svc:dual': 'False',
         'classifier:liblinear_svc:fit_intercept': 'True',
         'classifier:liblinear_svc:intercept_scaling': 1,
         'classifier:liblinear_svc:loss': 'squared_hinge',
         'classifier:liblinear_svc:multi_class': 'ovr',
         'classifier:liblinear_svc:penalty': 'l2',
         'classifier:liblinear_svc:tol': 0.00359411438055,
         'imputation:strategy': 'mean',
         'one_hot_encoding:minimum_fraction': 0.0018636449908690695,
         'one_hot_encoding:use_minimum_fraction': 'True',
         'preprocessor:__choice__': 'nystroem_sampler',
         'preprocessor:nystroem_sampler:kernel': 'cosine',
         'preprocessor:nystroem_sampler:n_components': 5183,
         'rescaling:__choice__': 'normalize'})),
     ]

targets = []
predictions = []
predictions_valid = []
predictions_test = []

# Make predictions and weight them
iteration = 0
for weight, classifier in choices:
    iteration += 1
    print(dataset, "Iteration %d/%d" % (iteration, len(choices)))
    try:
        classifier.fit(X.copy(), y.copy())
        predictions_valid.append(
            classifier.predict_proba(X_valid.copy()) * weight)
        predictions_test.append(
            classifier.predict_proba(X_test.copy()) * weight)
    except Exception as e:
        print(e)
        print(classifier.configuration)

# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0).astype(np.float32)

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ', fmt = '%.4e')
