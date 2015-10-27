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
dataset = 'madeline'
output = args.output

D = autosklearn.data.data_manager.DataManager(dataset, input)
X = D.data['X_train']
y = D.data['Y_train']
X_valid = D.data['X_valid']
X_test = D.data['X_test']

# Subset of features found with RFE. Feature with least importance in sklearn
# RF removed. Afterwards, trained RF on remaining features with 5CV. In the
# end, choose feature set with lowest error
features = [52, 70, 74, 83, 85, 135, 162, 183, 184, 185, 191, 197, 232, 237,
            239, 252]

X = X[:, features]
X_valid = X_valid[:, features]
X_test = X_test[:, features]

# Weights of the ensemble members as determined by Ensemble Selection
weights = np.array([0.100000, 0.080000, 0.080000, 0.060000, 0.060000,
                    0.060000, 0.060000, 0.040000, 0.040000, 0.040000,
                    0.040000, 0.040000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
                    0.020000, 0.020000])

# Ensemble members found by SMAC
configurations = [
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'median',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '4.0',
     'k_nearest_neighbors:p': '1.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'standard',
     'select_rates:alpha': '0.124513266268',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'weighting',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.802981892271',
     'kitchen_sinks:n_components': '704.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '7.66537661987',
     'qda:tol': '0.000779904033875',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.658527701661',
     'kitchen_sinks:n_components': '499.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '4.13193776587',
     'qda:tol': '0.0026677961139',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.658527701661',
     'kitchen_sinks:n_components': '498.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '7.39545021165',
     'qda:tol': '0.00116251661342',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.758771699267',
     'kitchen_sinks:n_components': '794.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '4.57263430441',
     'qda:tol': '0.00284918317943',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'most_frequent',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '5.0',
     'k_nearest_neighbors:p': '1.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'min/max',
     'select_rates:alpha': '0.0683198728939',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.773869494191',
     'kitchen_sinks:n_components': '608.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '5.34388968302',
     'qda:tol': '0.000118437687463',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'mean',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '4.0',
     'k_nearest_neighbors:p': '1.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'min/max',
     'select_rates:alpha': '0.0953909302386',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'chi2'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.722743897655',
     'kitchen_sinks:n_components': '952.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '3.61200930387',
     'qda:tol': '0.000911935213882',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'most_frequent',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '3.0',
     'k_nearest_neighbors:p': '2.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'standard',
     'select_rates:alpha': '0.12499749257',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '0.521009778754',
     'kitchen_sinks:n_components': '581.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '0.570532656005',
     'qda:tol': '0.00759604479274',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '0.736334496442',
     'kitchen_sinks:n_components': '590.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '8.78913455152',
     'qda:tol': '0.0417125881025',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'median',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '10.0',
     'k_nearest_neighbors:p': '2.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'min/max',
     'select_rates:alpha': '0.065583595323',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.725282605688',
     'kitchen_sinks:n_components': '591.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '4.32023431675',
     'qda:tol': '2.95483713232e-05',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.686955501206',
     'kitchen_sinks:n_components': '646.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '9.58493774318',
     'qda:tol': '0.00612419830773',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'median',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '6.0',
     'k_nearest_neighbors:p': '2.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'min/max',
     'select_rates:alpha': '0.276130352686',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'most_frequent',
     'kitchen_sinks:gamma': '0.549862378472',
     'kitchen_sinks:n_components': '591.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '1.11536443906',
     'qda:tol': '4.98941924261e-05',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '0.551878628115',
     'kitchen_sinks:n_components': '913.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '2.80643663684',
     'qda:tol': '0.0030955537468',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.797948222068',
     'kitchen_sinks:n_components': '856.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '0.753439507859',
     'qda:tol': '0.000179635997544',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'median',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '6.0',
     'k_nearest_neighbors:p': '2.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'standard',
     'select_rates:alpha': '0.121674691962',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'median',
     'kitchen_sinks:gamma': '0.870787144807',
     'kitchen_sinks:n_components': '591.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '3.25265485261',
     'qda:tol': '0.000232802336471',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.725282605688',
     'kitchen_sinks:n_components': '469.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '4.32023431675',
     'qda:tol': '6.11461737038e-05',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.742290491524',
     'kitchen_sinks:n_components': '699.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '1.80605719583',
     'qda:tol': '0.00759903394814',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'mean',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '4.0',
     'k_nearest_neighbors:p': '2.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'min/max',
     'select_rates:alpha': '0.0556366440458',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.69436212216',
     'kitchen_sinks:n_components': '477.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '7.19343875838',
     'qda:tol': '0.00130430743783',
     'rescaling:strategy': 'standard'},
    {'balancing:strategy': 'weighting',
     'classifier': 'k_nearest_neighbors',
     'imputation:strategy': 'median',
     'k_nearest_neighbors:algorithm': 'auto',
     'k_nearest_neighbors:leaf_size': '30.0',
     'k_nearest_neighbors:n_neighbors': '8.0',
     'k_nearest_neighbors:p': '1.0',
     'k_nearest_neighbors:weights': 'distance',
     'preprocessor': 'select_rates',
     'rescaling:strategy': 'standard',
     'select_rates:alpha': '0.0962781949808',
     'select_rates:mode': 'fdr',
     'select_rates:score_func': 'f_classif'},
    {'balancing:strategy': 'none',
     'classifier': 'qda',
     'imputation:strategy': 'mean',
     'kitchen_sinks:gamma': '0.680526800011',
     'kitchen_sinks:n_components': '627.0',
     'preprocessor': 'kitchen_sinks',
     'qda:reg_param': '3.3758872613',
     'qda:tol': '0.0025551077682',
     'rescaling:strategy': 'standard'},
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