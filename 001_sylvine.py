import argparse
import os

import numpy as np

import autosklearn
import autosklearn.data
import autosklearn.data.data_manager
import autosklearn.models.evaluator
from ParamSklearn.classification import ParamSklearnClassifier


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

input = args.input
dataset = 'sylvine'
output = args.output

D = autosklearn.data.data_manager.DataManager(dataset, input)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Subset of features found with RFE. Feature with least importance in sklearn
# RF removed. Afterwards, trained RF on remaining features with 5CV. In the
# end, choose feature set with lowest error
features = [6, 8, 9, 14]

X = X[:, features]
X_valid = X_valid[:, features]
X_test = X_test[:, features]

# Weights of the ensemble members as determined by Ensemble Selection
weights = np.array([0.420000, 0.360000, 0.060000, 0.040000, 0.040000,
                    0.040000, 0.020000, 0.020000])

# Ensemble members found by SMAC
configurations = [
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '1.92120672046',
     'kitchen_sinks:n_components': '716.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '1.58062868571',
     'qda:tol': '0.0247837474409',
     'rescaling:strategy': 'standard', },
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '1.61329137115',
     'kitchen_sinks:n_components': '500.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '5.45636866541',
     'qda:tol': '5.69425859943e-05',
     'rescaling:strategy': 'min/max', },
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '1.95127135806',
     'kitchen_sinks:n_components': '564.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '0.512205857283',
     'qda:tol': '0.000168304749916',
     'rescaling:strategy': 'standard', },
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '1.8592926955',
     'kitchen_sinks:n_components': '539.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '7.384724657',
     'qda:tol': '0.0200780040497',
     'rescaling:strategy': 'standard', },
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '0.968569589575',
     'kitchen_sinks:n_components': '528.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '5.73540397488',
     'qda:tol': '0.00632432527713',
     'rescaling:strategy': 'min/max', },
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '1.7159380388',
     'kitchen_sinks:n_components': '586.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '4.84995966137',
     'qda:tol': '0.0143521983037',
     'rescaling:strategy': 'standard', },
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'nystroem_sampler:gamma': '3.79316084659',
     'nystroem_sampler:kernel': 'rbf',
     'nystroem_sampler:n_components': '516.0',
     'preprocessor': 'nystroem_sampler',
     'qda:reg_param': '9.63571710058',
     'qda:tol': '0.00901955088569',
     'rescaling:strategy': 'min/max', },
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '1.85336603609',
     'kitchen_sinks:n_components': '509.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '8.57076337966',
     'qda:tol': '0.000361249119707',
     'rescaling:strategy': 'standard'}
]

classifiers = []
predictions_valid = []
predictions_test = []

# Make predictions and weight them
for weight, configuration in zip(weights, configurations):
    for param in configuration:
        try:
            configuration[param] = int(configuration[param])
        except Exception:
            try:
                configuration[param] = float(configuration[param])
            except Exception:
                pass

    classifier = ParamSklearnClassifier(configuration, 1)
    classifiers.append(classifier)
    try:
        classifier.fit(X.copy(), y.copy())
        predictions_valid.append(
            classifier.predict_proba(X_valid.copy()) * weight)
        predictions_test.append(
            classifier.predict_proba(X_test.copy()) * weight)
    except Exception as e:
        print e
        print configuration

# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0)
    predictions = predictions[:,1].reshape((-1, 1))

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ')