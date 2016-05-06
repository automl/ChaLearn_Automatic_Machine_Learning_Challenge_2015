import numpy as np
import scipy.sparse as sp

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class RegDeepNet(AutoSklearnRegressionAlgorithm):

    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 dropout_layer_1, dropout_output, std_layer_1,
                 learning_rate, solver, lambda2, activation,
                 num_units_layer_2=10, num_units_layer_3=10, num_units_layer_4=10,
                 num_units_layer_5=10, num_units_layer_6=10,
                 dropout_layer_2=0.5, dropout_layer_3=0.5, dropout_layer_4=0.5,
                 dropout_layer_5=0.5, dropout_layer_6=0.5,
                 std_layer_2=0.005, std_layer_3=0.005, std_layer_4=0.005,
                 std_layer_5=0.005, std_layer_6=0.005,
                 momentum=0.99, beta1=0.9, beta2=0.9, rho=0.95,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=2,
                 random_state=None):
        self.number_updates = number_updates
        self.batch_size = batch_size
        # Hacky implementation of condition on number of layers
        self.num_layers = ord(num_layers) - ord('a')
        self.dropout_output = dropout_output
        self.learning_rate = learning_rate
        self.lr_policy = lr_policy
        self.lambda2 = lambda2
        self.momentum = momentum
        self.beta1 = 1-beta1
        self.beta2 = 1-beta2
        self.rho = rho
        self.solver = solver
        self.activation = activation
        self.gamma = gamma
        self.power = power
        self.epoch_step = epoch_step

        # Empty features and shape
        self.n_features = None
        self.input_shape = None
        self.m_issparse = False
        self.m_isregression = True

        # To avoid eval call. Could be done with **karws
        args = locals()

        self.num_units_per_layer = []
        self.dropout_per_layer = []
        self.std_per_layer = []
        for i in range(1, self.num_layers):
            self.num_units_per_layer.append(int(args.get("num_units_layer_" + str(i))))
            self.dropout_per_layer.append(float(args.get("dropout_layer_" + str(i))))
            self.std_per_layer.append(float(args.get("std_layer_" + str(i))))
        self.estimator = None

    def _prefit(self, X, y):
        self.batch_size = int(self.batch_size)
        self.n_features = X.shape[1]
        self.input_shape = (self.batch_size, self.n_features)

        assert len(self.num_units_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"
        assert len(self.dropout_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"

        self.num_output_units = 1  # Regression
        # Normalize the output - Suggestion on 24.04
        self.mean_y = np.mean(y)
        self.std_y = np.std(y)
        y = (y - self.mean_y) / self.std_y
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        self.m_issparse = sp.issparse(X)

        return X, y

    def fit(self, X, y):

        Xf, yf = self._prefit(X, y)

        epoch = (self.number_updates * self.batch_size)//X.shape[0]
        number_epochs = min(max(2, epoch), 50)  # Cap the max number of possible epochs

        from ...implementations import FeedForwardNet
        self.estimator = FeedForwardNet.FeedForwardNet(batch_size=self.batch_size,
                                                       input_shape=self.input_shape,
                                                       num_layers=self.num_layers,
                                                       num_units_per_layer=self.num_units_per_layer,
                                                       dropout_per_layer=self.dropout_per_layer,
                                                       std_per_layer=self.std_per_layer,
                                                       num_output_units=self.num_output_units,
                                                       dropout_output=self.dropout_output,
                                                       learning_rate=self.learning_rate,
                                                       lr_policy=self.lr_policy,
                                                       lambda2=self.lambda2,
                                                       momentum=self.momentum,
                                                       beta1=self.beta1,
                                                       beta2=self.beta2,
                                                       rho=self.rho,
                                                       solver=self.solver,
                                                       activation=self.activation,
                                                       num_epochs=number_epochs,
                                                       gamma=self.gamma,
                                                       power=self.power,
                                                       epoch_step=self.epoch_step,
                                                       is_sparse=self.m_issparse,
                                                       is_binary=False,
                                                       is_regression=self.m_isregression)
        self.estimator.fit(Xf, yf)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        preds = self.estimator.predict(X, self. m_issparse)
        return preds * self.std_y + self.mean_y

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X, self.m_issparse)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'feed_nn',
                'name': 'Feed Forward Neural Network',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # GPUTRACK: Based on http://svail.github.io/rnn_perf/
        # We make batch size and number of units multiples of 64

        # Hacky way to condition layers params based on the number of layers
        # GPUTRACK: Reduced number of layers
        # 'c'=1, 'd'=2, 'e'=3 ,'f'=4 + output_layer
        # layer_choices = [chr(i) for i in xrange(ord('c'), ord('e'))]

        layer_choices = ['c', 'd', 'e']

        batch_size = UniformIntegerHyperparameter("batch_size",
                                                  64, 2048,
                                                  default=550)

        number_updates = UniformIntegerHyperparameter("number_updates",
                                                      200, 5500,
                                                      log=True,
                                                      default=512)

        num_layers = CategoricalHyperparameter("num_layers",
                                               choices=layer_choices,
                                               default='c')

        num_units_layer_1 = UniformIntegerHyperparameter("num_units_layer_1",
                                                         64, 4096,
                                                         default=128)

        num_units_layer_2 = UniformIntegerHyperparameter("num_units_layer_2",
                                                         64, 4096,
                                                         default=128)
        num_units_layer_3 = UniformIntegerHyperparameter("num_units_layer_3",
                                                         64, 4096,
                                                         log=True,
                                                         default=128)

        dropout_layer_1 = UniformFloatHyperparameter("dropout_layer_1",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_2 = UniformFloatHyperparameter("dropout_layer_2",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_3 = UniformFloatHyperparameter("dropout_layer_3",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_output = UniformFloatHyperparameter("dropout_output",
                                                    0.0, 0.99,
                                                    default=0.5)

        lr = CategoricalHyperparameter("learning_rate",
                                       choices=[1e-1, 1e-2, 1e-3, 1e-4],
                                       default=1e-2)

        l2 = UniformFloatHyperparameter("lambda2", 1e-6, 1e-2, log=True,
                                        default=1e-3)

        std_layer_1 = UniformFloatHyperparameter("std_layer_1", 0.001, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_2 = UniformFloatHyperparameter("std_layer_2", 0.001, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_3 = UniformFloatHyperparameter("std_layer_3", 0.001, 0.1,
                                                 log=True,
                                                 default=0.005)

        # Using Tobias' adam
        solver = Constant(name="solver", value="smorm3s")

        non_linearities = CategoricalHyperparameter(name='activation',
                                                    choices=['tanh', 'scaledTanh', 'sigmoid'],
                                                    default='tanh')

        cs = ConfigurationSpace()
        # cs.add_hyperparameter(number_epochs)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(num_units_layer_1)
        cs.add_hyperparameter(num_units_layer_2)
        cs.add_hyperparameter(num_units_layer_3)
        cs.add_hyperparameter(dropout_layer_1)
        cs.add_hyperparameter(dropout_layer_2)
        cs.add_hyperparameter(dropout_layer_3)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(std_layer_1)
        cs.add_hyperparameter(std_layer_2)
        cs.add_hyperparameter(std_layer_3)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)
        cs.add_hyperparameter(solver)
        cs.add_hyperparameter(non_linearities)

        layer_2_condition = InCondition(num_units_layer_2, num_layers,
                                        ['d', 'e'])
        layer_3_condition = InCondition(num_units_layer_3, num_layers,
                                        ['e'])
        cs.add_condition(layer_2_condition)
        cs.add_condition(layer_3_condition)

        # Condition dropout parameter on layer choice
        dropout_2_condition = InCondition(dropout_layer_2, num_layers,
                                          ['d', 'e'])
        dropout_3_condition = InCondition(dropout_layer_3, num_layers,
                                          ['e'])
        cs.add_condition(dropout_2_condition)
        cs.add_condition(dropout_3_condition)

        # Condition std parameter on layer choice
        std_2_condition = InCondition(std_layer_2, num_layers, ['d', 'e'])
        std_3_condition = InCondition(std_layer_3, num_layers, ['e'])
        cs.add_condition(std_2_condition)
        cs.add_condition(std_3_condition)

        return cs
