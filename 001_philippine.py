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
dataset = 'philippine'
output = args.output

D = autosklearn.data.data_manager.DataManager(dataset, input)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Subset of features found with RFE. Feature with least importance in sklearn
# RF removed. Afterwards, trained RF on remaining features with 5CV. In the
# end, choose feature set with lowest error
features = [33, 89, 140, 168, 178, 271]

X = X[:, features]
X_valid = X_valid[:, features]
X_test = X_test[:, features]

# Weights of the ensemble members as determined by Ensemble Selection
weights = np.array([0.100000, 0.080000, 0.080000, 0.060000, 0.040000,
                    0.040000, 0.040000, 0.040000, 0.040000, 0.040000,
                    0.040000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000])

# Ensemble members found by SMAC
configurations = [
    {'adaboost:algorithm': 'SAMME.R',
     'adaboost:learning_rate': '0.243038132773',
     'adaboost:max_depth': '9.0',
     'adaboost:n_estimators': '475.0',
     'balancing:strategy': 'none',
     'classifier': 'adaboost',
     'feature_agglomeration:affinity': 'cosine',
     'feature_agglomeration:linkage': 'complete',
     'feature_agglomeration:n_clusters': '287.0',
     'imputation:strategy': 'most_frequent',
     'preprocessor': 'feature_agglomeration',
     'rescaling:strategy': 'none',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.246430392425',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '436.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '156.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.205679811363',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '485.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '79.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.250841964136',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '479.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '352.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'none',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.329040651125',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '493.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '268.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.376704790019',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '400.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'ward',
        'feature_agglomeration:n_clusters': '344.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.483824181899',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '479.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '310.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.246430392425',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '494.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '156.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.319596208353',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '446.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '65.0',
        'imputation:strategy': 'mean',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.208071429428',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '487.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '219.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'none',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.362379903949',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '389.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '123.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.468508930474',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '477.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '244.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.284273806405',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '483.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '174.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.2635286978',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '482.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '118.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.326966274076',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '494.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '87.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.239427049389',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '393.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '331.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.272345990341',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '478.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '20.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.36300772469',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '430.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '88.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.29318612753',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '418.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '220.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.315769388471',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '494.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '270.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.295544282435',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '478.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '195.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.298219714131',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '473.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '39.0',
        'imputation:strategy': 'mean',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.370877623224',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '382.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '331.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.339058617161',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '466.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '38.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.272345990341',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '478.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'cosine',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '68.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'none',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.268568387674',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '499.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '78.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'standard',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.286357615604',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '490.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'ward',
        'feature_agglomeration:n_clusters': '220.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.377112372612',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '458.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'ward',
        'feature_agglomeration:n_clusters': '125.0',
        'imputation:strategy': 'most_frequent',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.400954561452',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '408.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'euclidean',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '345.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.196044249482',
        'adaboost:max_depth': '9.0',
        'adaboost:n_estimators': '494.0',
        'balancing:strategy': 'none',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'average',
        'feature_agglomeration:n_clusters': '182.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'min/max',
    },
    {
        'adaboost:algorithm': 'SAMME.R',
        'adaboost:learning_rate': '0.312315129765',
        'adaboost:max_depth': '10.0',
        'adaboost:n_estimators': '442.0',
        'balancing:strategy': 'weighting',
        'classifier': 'adaboost',
        'feature_agglomeration:affinity': 'manhattan',
        'feature_agglomeration:linkage': 'complete',
        'feature_agglomeration:n_clusters': '347.0',
        'imputation:strategy': 'median',
        'preprocessor': 'feature_agglomeration',
        'rescaling:strategy': 'none'}
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