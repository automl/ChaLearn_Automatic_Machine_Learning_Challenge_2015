import argparse
import os

import numpy as np
import sklearn.cross_validation

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
dataset = 'christine'
output = args.output

D = autosklearn.data.data_manager.DataManager(dataset, input)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

weights = np.array([1.0])

# Choosing the single best model without feature selection by RFE (but by
# select percentile classification which is in the auto-sklearn pipeline) seems
# to work best here
configurations = [
    {'balancing:strategy': 'none',
     'classifier': 'libsvm_svc',
     'imputation:strategy': 'median',
     'libsvm_svc:C': '5.06888516101',
     'libsvm_svc:class_weight': 'None',
     'libsvm_svc:gamma': '0.0870955322069',
     'libsvm_svc:kernel': 'rbf',
     'libsvm_svc:max_iter': '-1.0',
     'libsvm_svc:shrinking': 'False',
     'libsvm_svc:tol': '2.62849564978e-05',
     'preprocessor': 'select_percentile_classification',
     'rescaling:strategy': 'min/max',
     'select_percentile_classification:percentile': '36.4058569521',
     'select_percentile_classification:score_func': 'f_classif'}
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
    predictions = predictions[:, 1].reshape((-1, 1))

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ')