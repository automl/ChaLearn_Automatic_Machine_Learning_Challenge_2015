import os
import cPickle
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T
import six


import autosklearn.data.competition_data_manager


class FeedForwardNet(object):

    def __init__(self, input_shape=(1, 28, 28),
                 batch_size=100,
                 num_layers=4,
                 num_units_per_layer=(10, 10, 10),
                 dropout_per_layer=(0.5, 0.5, 0.5),
                 std_per_layer=(0.1, 0.1, 0.1),
                 num_output_units=10,
                 dropout_output=0.5,
                 learning_rate=0.01,
                 momentum=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 rho=0.95,
                 solver="sgd"):

        assert len(num_units_per_layer) == num_layers - 1
        assert len(dropout_per_layer) == num_layers - 1

        self.batch_size = int(batch_size)
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        print("... building network")
        print(input_shape)
        self.network = lasagne.layers.InputLayer(shape=input_shape,
                                                input_var=input_var)

        # Define each layer
        for i in range(num_layers - 1):
            self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                p=float(dropout_per_layer[i])),
                 num_units=int(num_units_per_layer[i]),
                 W=lasagne.init.Normal(std=float(std_per_layer[i]), mean=0),
                 b=lasagne.init.Constant(val=0.0),
                 nonlinearity=lasagne.nonlinearities.rectify)

        # Define output layer
        self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network, p=float(dropout_output)),
                 num_units=int(num_output_units),
                 W=lasagne.init.GlorotNormal(),
                 b=lasagne.init.Constant(),
                 nonlinearity=lasagne.nonlinearities.softmax)

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        loss = loss.mean()

        self.learning_rate = theano.shared(np.array(learning_rate,
                                                dtype=theano.config.floatX))

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        if solver == "nesterov":
            updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=learning_rate,
                                                momentum=momentum)
        elif solver == "adam":
            updates = lasagne.updates.adam(loss, params,
                                           learning_rate=learning_rate,
                                           beta1=beta1, beta2=beta2)
        elif solver == "adadelta":
            updates = lasagne.updates.adadelta(loss, params,
                                               learning_rate=learning_rate,
                                               rho=rho)
        elif solver == "adagrad":
            updates = lasagne.updates.adagrad(loss, params,
                                              learning_rate=learning_rate)
        elif solver == "sgd":
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=learning_rate)
        elif solver == "momentum":
            updates = lasagne.updates.momentum(loss, params,
                                               learning_rate=learning_rate,
                                               momentum=momentum)
        else:
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=learning_rate)
        valid_prediction = lasagne.layers.get_output(self.network,
                                                     deterministic=True)
        valid_loss = lasagne.objectives.categorical_crossentropy(
                                                        valid_prediction,
                                                        target_var)
        valid_loss = valid_loss.mean()
        valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1),
                                target_var),
                                dtype=theano.config.floatX)

        print("... compiling theano functions")
        self.train_fn = theano.function([input_var, target_var], loss,
                                        updates=updates,
                                        allow_input_downcast=True)
        self.val_fn = theano.function([input_var, target_var],
                                      [valid_loss, valid_acc],
                                      allow_input_downcast=True)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def train(model, X_train, y_train, X_valid, y_valid, num_epochs=500,
          lr_policy="inv", gamma=0.1, power=0.5, step=30):

    learning_curve = np.zeros([num_epochs])

    lr_base = model.learning_rate.get_value()
    for epoch in range(num_epochs):

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, model.batch_size, shuffle=True):
            inputs, targets = batch
            train_err += model.train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        # Learning rate adaption
        if lr_policy == "inv":
            decay = np.power((1 + gamma * epoch), (- power))
        elif lr_policy == "exp":
            decay = np.power(gamma, epoch)
        elif lr_policy == "step":
            decay = np.power(gamma, (np.floor(epoch / float(step))))
        elif lr_policy == "fixed":
            decay = 1

        decay = np.array([decay], dtype=np.float32)
        print ("decay factor %f" % decay)
        model.learning_rate.set_value(lr_base * decay)
        print ("Learning rate %f" % model.learning_rate.get_value())
        for batch in iterate_minibatches(X_valid, y_valid, model.batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err, acc = model.val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        learning_curve[epoch] = (val_acc / float(val_batches) * 100)

    return 1 - (val_acc / val_batches), learning_curve


dataset = 'alexis'

output = "/home/kleinaa/experiments/automated_deep_learning/automl/alexis/reproduction"
path = os.path.join("/data/aad/automl_data/003", dataset)

dm = autosklearn.data.competition_data_manager.CompetitionDataManager(path)
X_train = dm.data["X_train"].toarray()
X_train = np.array(X_train, dtype=np.float32)
Y_train = np.argmax(np.array(dm.data["Y_train"], dtype=np.float32), axis=1)
X_valid = np.array(dm.data["X_valid"].toarray(), dtype=np.float32)
X_test = np.array(dm.data["X_test"].toarray(), dtype=np.float32)

# Replace the following array by a new ensemble
choices = \
    [({'num_layers': 3.0,
       'power': 0.9608086215808905,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.32350553193868387,
       'dropout_output': 0.19522565771560596,
       'learning_rate': 0.22087420991966428,
       'dropout_layer_2': 0.5416862005436668,
       'num_units_layer_2': 3429.0,
       'batch_size': 963.3316054786878,
       'num_units_layer_1': 4742.0,
       'step': 17.0,
       'momentum': 0.8358157637992942,
       'num_epochs': 490.0,
       'std_layer_2': 0.002570301444272789,
       'lr_policy': 'fixed',
       'std_layer_1': 0.00203675415789638,
       'gamma': 0.1982457569106505}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.9543822789540591,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.32350553193868387,
       'dropout_output': 0.19522565771560596,
       'dropout_layer_3': 0.07073518892061528,
       'dropout_layer_2': 0.5416862005436668,
       'num_units_layer_2': 3429.0,
       'num_units_layer_3': 2921.0,
       'batch_size': 963.3316054786878,
       'num_units_layer_1': 4742.0,
       'step': 17.0,
       'std_layer_1': 0.00203675415789638,
       'momentum': 0.8358157637992942,
       'num_epochs': 490.0,
       'std_layer_3': 0.0027860724140478716,
       'std_layer_2': 0.002570301444272789,
       'lr_policy': 'step',
       'learning_rate': 0.22087420991966428,
       'gamma': 0.1982457569106505}, 0.02),
     ({'dropout_output': 0.2219836761125169,
       'num_units_layer_4': 503.0,
       'num_epochs': 249.0,
       'num_units_layer_2': 1798.0,
       'num_units_layer_3': 4435.0,
       'num_units_layer_1': 5785.0,
       'num_layers': 5.0,
       'beta2': 0.35170616682914413,
       'beta1': 0.3871526867361012,
       'std_layer_3': 0.0020693367447187833,
       'std_layer_2': 0.01626605159391926,
       'std_layer_1': 0.010029613006349311,
       'learning_rate': 0.161203081475512,
       'std_layer_4': 0.04266796418763915,
       'power': 0.7730778282076188,
       'dropout_layer_1': 0.5996870640229259,
       'dropout_layer_3': 0.016146856157454967,
       'dropout_layer_2': 0.049116767959870425,
       'dropout_layer_4': 0.6672322817450616,
       'batch_size': 741.3580222134027,
       'step': 21.0,
       'lr_policy': 'fixed',
       'solver': '"adam"',
       'gamma': 0.6788656698879015}, 0.040000000000000001),
     ({'num_layers': 4.0,
       'power': 0.7949015212265006,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.4521497425726368,
       'dropout_output': 0.44804264005915573,
       'dropout_layer_3': 0.5015510137781847,
       'dropout_layer_2': 0.45385947764882123,
       'num_units_layer_2': 5059.0,
       'num_units_layer_3': 291.0,
       'batch_size': 808.4213834239384,
       'num_units_layer_1': 2468.0,
       'step': 39.0,
       'std_layer_1': 0.0014528596407236882,
       'rho': 0.5198404112822554,
       'num_epochs': 365.0,
       'std_layer_3': 0.018883281420906187,
       'std_layer_2': 0.060173127055786896,
       'lr_policy': 'inv',
       'learning_rate': 0.004445343676392594,
       'gamma': 0.5816489765526873}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.22582421055308477,
       'learning_rate': 0.19202636600670817,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 4648.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 76.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'fixed',
       'std_layer_1': 0.012567729774967713,
       'gamma': 0.6722873907885709}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.3357527566089538,
       'learning_rate': 0.19202636600670817,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 6138.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 68.0,
       'rho': 0.6568839756050975,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'fixed',
       'std_layer_1': 0.012567729774967713,
       'gamma': 0.6722873907885709}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.8196900527717096,
       'solver': '"sgd"',
       'dropout_layer_1': 0.2939609278437122,
       'dropout_output': 0.2553598975807717,
       'learning_rate': 0.0792359335703241,
       'dropout_layer_2': 0.17148792984538072,
       'num_units_layer_2': 6042.0,
       'batch_size': 68.01704776734499,
       'num_units_layer_1': 4337.0,
       'step': 37.0,
       'momentum': 0.7768588564816645,
       'num_epochs': 467.0,
       'std_layer_2': 0.09305184343463795,
       'lr_policy': 'inv',
       'std_layer_1': 0.0033240180436903202,
       'gamma': 0.7240342867167526}, 0.040000000000000001),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.7836649078292658,
       'dropout_output': 0.21660605836335647,
       'learning_rate': 0.027610542766288576,
       'dropout_layer_2': 0.011586976261760496,
       'num_units_layer_2': 5787.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 4722.0,
       'step': 18.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'exp',
       'std_layer_1': 0.0005598439981691678,
       'gamma': 0.9667180658562067}, 0.040000000000000001),
     ({'num_layers': 3.0,
       'power': 0.9747702817998765,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.4051640090922492,
       'dropout_output': 0.2892451691812821,
       'learning_rate': 0.17394384345131372,
       'dropout_layer_2': 0.5231272729308043,
       'num_units_layer_2': 5608.0,
       'batch_size': 73.2292167635178,
       'num_units_layer_1': 4633.0,
       'step': 16.0,
       'momentum': 0.8358157637992942,
       'num_epochs': 491.0,
       'std_layer_2': 0.002570301444272789,
       'lr_policy': 'step',
       'std_layer_1': 0.00696310505976957,
       'gamma': 0.1982457569106505}, 0.040000000000000001),
     ({'dropout_output': 0.25306857146792566,
       'num_units_layer_4': 4733.0,
       'num_epochs': 486.0,
       'num_units_layer_2': 5797.0,
       'num_units_layer_3': 5620.0,
       'num_units_layer_1': 4722.0,
       'num_layers': 5.0,
       'std_layer_3': 0.0018624928976982682,
       'std_layer_2': 0.012236381842172813,
       'std_layer_1': 0.0013146995716357634,
       'learning_rate': 0.0528394424911447,
       'std_layer_4': 0.030949659698034787,
       'power': 0.951501985775478,
       'dropout_layer_1': 0.2602384986628831,
       'dropout_layer_3': 0.07918985321674664,
       'dropout_layer_2': 0.13607828166784158,
       'dropout_layer_4': 0.20468440586381545,
       'batch_size': 118.70898813118156,
       'step': 18.0,
       'rho': 0.6568839756050975,
       'lr_policy': 'exp',
       'solver': '"adadelta"',
       'gamma': 0.7090475445273005}, 0.040000000000000001),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.3795818048708614,
       'learning_rate': 0.19202636600670817,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 4648.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 76.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'fixed',
       'std_layer_1': 0.00019210721668567557,
       'gamma': 0.6722873907885709}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"sgd"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.3357527566089538,
       'learning_rate': 0.19202636600670817,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 6138.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 68.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'fixed',
       'std_layer_1': 0.020725423936460784,
       'gamma': 0.6722873907885709}, 0.02),
     ({'dropout_output': 0.5544528745031847,
       'num_units_layer_4': 4835.0,
       'num_units_layer_5': 1207.0,
       'num_units_layer_2': 3532.0,
       'num_units_layer_3': 3265.0,
       'num_units_layer_1': 2341.0,
       'num_epochs': 79.0,
       'std_layer_3': 0.0003041749397972243,
       'std_layer_2': 0.015965838945860733,
       'lr_policy': 'step',
       'learning_rate': 0.003930848214214284,
       'momentum': 0.5508087063095997,
       'std_layer_4': 0.042866542378616,
       'power': 0.7392683450424904,
       'dropout_layer_1': 0.09142524602061186,
       'dropout_layer_3': 0.45551111081428075,
       'dropout_layer_2': 0.6039763657544621,
       'dropout_layer_5': 0.4194002794764725,
       'dropout_layer_4': 0.15409260770878827,
       'batch_size': 913.9939922328252,
       'step': 9.0,
       'std_layer_1': 0.018242827552014805,
       'num_layers': 6.0,
       'solver': '"momentum"',
       'std_layer_5': 0.080768263392991,
       'gamma': 0.12406835380828729}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.3795818048708614,
       'learning_rate': 0.28665652540783937,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 4648.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 76.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'exp',
       'std_layer_1': 0.025068182392771483,
       'gamma': 0.6722873907885709}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.9501928234008037,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.49417282390370604,
       'dropout_output': 0.1112655122924812,
       'learning_rate': 0.05647348442267938,
       'dropout_layer_2': 0.5668268697567368,
       'num_units_layer_2': 847.0,
       'batch_size': 69.43044908879082,
       'num_units_layer_1': 4509.0,
       'step': 62.0,
       'rho': 0.1412178971508795,
       'num_epochs': 325.0,
       'std_layer_2': 0.09245714688025959,
       'lr_policy': 'inv',
       'std_layer_1': 0.0030031803285322065,
       'gamma': 0.9880106805712771}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.9982859561242945,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3868459283104136,
       'dropout_output': 0.30578911000133563,
       'dropout_layer_3': 0.05910805094837215,
       'dropout_layer_2': 0.5346257876302651,
       'num_units_layer_2': 1555.0,
       'num_units_layer_3': 3709.0,
       'batch_size': 793.5704826524803,
       'num_units_layer_1': 5632.0,
       'step': 91.0,
       'std_layer_1': 0.0019427866308684189,
       'momentum': 0.47096093223996227,
       'num_epochs': 488.0,
       'std_layer_3': 0.004455993940858878,
       'std_layer_2': 0.07300638871271027,
       'lr_policy': 'exp',
       'learning_rate': 0.1414121563748416,
       'gamma': 0.1338226652313088}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"sgd"',
       'dropout_layer_1': 0.3653306482580645,
       'dropout_output': 0.3357527566089538,
       'learning_rate': 0.19202636600670817,
       'dropout_layer_2': 0.21819303981244112,
       'num_units_layer_2': 6138.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 5766.0,
       'step': 68.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'fixed',
       'std_layer_1': 0.012567729774967713,
       'gamma': 0.6722873907885709}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.6102313152642265,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.5002438300235051,
       'dropout_output': 0.23534479317312496,
       'dropout_layer_3': 0.6828689388536276,
       'dropout_layer_2': 0.1762597535432976,
       'num_units_layer_2': 3647.0,
       'num_units_layer_3': 3044.0,
       'batch_size': 786.4476079179202,
       'num_units_layer_1': 5076.0,
       'step': 60.0,
       'std_layer_1': 0.013221036066045333,
       'momentum': 0.9814146239572572,
       'num_epochs': 478.0,
       'std_layer_3': 0.07660361796208646,
       'std_layer_2': 0.021796162268579933,
       'lr_policy': 'step',
       'learning_rate': 0.20931601325873997,
       'gamma': 0.29777339200275743}, 0.02),
     ({'dropout_output': 0.6986654447282298,
       'num_units_layer_4': 2724.0,
       'num_epochs': 70.0,
       'num_units_layer_2': 940.0,
       'num_units_layer_3': 176.0,
       'num_units_layer_1': 1554.0,
       'num_layers': 5.0,
       'beta2': 0.3437234235760527,
       'beta1': 0.5122595623529058,
       'std_layer_3': 0.001323462426190976,
       'std_layer_2': 0.08527672724824989,
       'std_layer_1': 0.06598031837899644,
       'learning_rate': 5.88849354815407e-05,
       'std_layer_4': 0.003244516882128532,
       'power': 0.8191416561238503,
       'dropout_layer_1': 0.12585693415903498,
       'dropout_layer_3': 0.2642128122930812,
       'dropout_layer_2': 0.07731697305608044,
       'dropout_layer_4': 0.016910080279172977,
       'batch_size': 803.0778892408711,
       'step': 82.0,
       'lr_policy': 'exp',
       'solver': '"adam"',
       'gamma': 0.9734274599579855}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.9839161318019856,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.3286158737442102,
       'dropout_output': 0.2602697774913525,
       'learning_rate': 0.05647348442267938,
       'dropout_layer_2': 0.41223774560421556,
       'num_units_layer_2': 3119.0,
       'batch_size': 69.43044908879082,
       'num_units_layer_1': 4509.0,
       'step': 62.0,
       'rho': 0.1412178971508795,
       'num_epochs': 341.0,
       'std_layer_2': 0.07883910009093228,
       'lr_policy': 'inv',
       'std_layer_1': 0.002283362184875592,
       'gamma': 0.9880106805712771}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.8211518930492289,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3288790841138845,
       'dropout_output': 0.2458740233193249,
       'learning_rate': 0.07112784792816951,
       'dropout_layer_2': 0.17005904025630333,
       'num_units_layer_2': 5787.0,
       'batch_size': 254.02504767576468,
       'num_units_layer_1': 4722.0,
       'step': 18.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'exp',
       'std_layer_1': 0.0005598439981691678,
       'gamma': 0.9667180658562067}, 0.02),
     ({'dropout_output': 0.6942222485334036,
       'num_units_layer_4': 4488.0,
       'num_epochs': 246.0,
       'num_units_layer_2': 360.0,
       'num_units_layer_3': 1762.0,
       'num_units_layer_1': 2452.0,
       'num_layers': 5.0,
       'std_layer_3': 0.000840782752525304,
       'std_layer_2': 0.053417714428946675,
       'std_layer_1': 2.49410460816067e-05,
       'learning_rate': 0.030725262843106595,
       'momentum': 0.7354775983645889,
       'std_layer_4': 0.014723488261112642,
       'power': 0.6825025080767805,
       'dropout_layer_1': 0.5430371683124818,
       'dropout_layer_3': 0.7178289875612346,
       'dropout_layer_2': 0.17679879849784874,
       'dropout_layer_4': 0.670500227550017,
       'batch_size': 848.7089176912435,
       'step': 15.0,
       'lr_policy': 'inv',
       'solver': '"nesterov"',
       'gamma': 0.6360825538930697}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.7084104721089539,
       'solver': '"adam"',
       'dropout_layer_1': 0.3228804321714691,
       'dropout_output': 0.19123621059916907,
       'dropout_layer_3': 0.5991741560861912,
       'dropout_layer_2': 0.042708968496055696,
       'num_units_layer_2': 4473.0,
       'num_units_layer_3': 299.0,
       'batch_size': 662.3881155284046,
       'num_units_layer_1': 5531.0,
       'step': 75.0,
       'std_layer_1': 0.006831817706575724,
       'beta2': 0.5763734437755985,
       'beta1': 0.46578559803031666,
       'num_epochs': 454.0,
       'std_layer_3': 0.07534205836424511,
       'std_layer_2': 0.010541138669952833,
       'lr_policy': 'step',
       'learning_rate': 0.2286213615447043,
       'gamma': 0.9847806583536367}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.6074047952114839,
       'solver': '"sgd"',
       'dropout_layer_1': 0.3837602436079431,
       'dropout_output': 0.2586093771445448,
       'learning_rate': 0.1850908533669142,
       'dropout_layer_2': 0.2012363291904858,
       'num_units_layer_2': 2999.0,
       'batch_size': 252.36804895745078,
       'num_units_layer_1': 4742.0,
       'step': 98.0,
       'momentum': 0.5716625593124078,
       'num_epochs': 417.0,
       'std_layer_2': 0.04035506633883424,
       'lr_policy': 'fixed',
       'std_layer_1': 0.03471403541293925,
       'gamma': 0.9228259167471723}, 0.02),
     ({'dropout_output': 0.012311474913025122,
       'num_units_layer_4': 5022.0,
       'num_epochs': 425.0,
       'num_units_layer_2': 1134.0,
       'num_units_layer_3': 5339.0,
       'num_units_layer_1': 1766.0,
       'num_layers': 5.0,
       'std_layer_3': 0.022935273875549143,
       'std_layer_2': 0.03211092898082504,
       'std_layer_1': 0.0009145066420452281,
       'learning_rate': 0.020646689616918727,
       'momentum': 0.5669637108787287,
       'std_layer_4': 0.0777074650891919,
       'power': 0.9413926402897054,
       'dropout_layer_1': 0.8245359519955847,
       'dropout_layer_3': 0.5298513101402106,
       'dropout_layer_2': 0.8202560119711875,
       'dropout_layer_4': 0.627756689182714,
       'batch_size': 986.6136297054668,
       'step': 46.0,
       'lr_policy': 'inv',
       'solver': '"momentum"',
       'gamma': 0.965503310296916}, 0.059999999999999998),
     ({'num_layers': 4.0,
       'power': 0.7084104721089539,
       'solver': '"adam"',
       'dropout_layer_1': 0.3228804321714691,
       'dropout_output': 0.19123621059916907,
       'dropout_layer_3': 0.5991741560861912,
       'dropout_layer_2': 0.042708968496055696,
       'num_units_layer_2': 3844.0,
       'num_units_layer_3': 299.0,
       'batch_size': 962.2936600774519,
       'num_units_layer_1': 5531.0,
       'step': 75.0,
       'std_layer_1': 0.0017111040466369133,
       'beta2': 0.5763734437755985,
       'beta1': 0.46578559803031666,
       'num_epochs': 475.0,
       'std_layer_3': 0.07534205836424511,
       'std_layer_2': 0.010541138669952833,
       'lr_policy': 'step',
       'learning_rate': 0.2286213615447043,
       'gamma': 0.9847806583536367}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.7084104721089539,
       'solver': '"adam"',
       'dropout_layer_1': 0.3228804321714691,
       'dropout_output': 0.19123621059916907,
       'learning_rate': 0.2286213615447043,
       'dropout_layer_2': 0.042708968496055696,
       'num_units_layer_2': 3844.0,
       'batch_size': 962.2936600774519,
       'num_units_layer_1': 5531.0,
       'step': 75.0,
       'beta2': 0.5763734437755985,
       'beta1': 0.46578559803031666,
       'num_epochs': 475.0,
       'std_layer_2': 0.010541138669952833,
       'lr_policy': 'step',
       'std_layer_1': 0.0017111040466369133,
       'gamma': 0.9847806583536367}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.6102829556412925,
       'solver': '"sgd"',
       'dropout_layer_1': 0.04382776539965009,
       'dropout_output': 0.4008248943831341,
       'learning_rate': 0.24435066375760794,
       'dropout_layer_2': 0.692660907091548,
       'num_units_layer_2': 1375.0,
       'batch_size': 427.01583837951085,
       'num_units_layer_1': 5102.0,
       'step': 79.0,
       'momentum': 0.6475139577982485,
       'num_epochs': 320.0,
       'std_layer_2': 0.07399922539485788,
       'lr_policy': 'step',
       'std_layer_1': 0.09262827614546214,
       'gamma': 0.7776717365467021}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.6419781172534461,
       'solver': '"momentum"',
       'dropout_layer_1': 0.9717726710481605,
       'dropout_output': 0.32908147435708823,
       'dropout_layer_3': 0.7055634222219525,
       'dropout_layer_2': 0.8546874464796694,
       'num_units_layer_2': 3851.0,
       'num_units_layer_3': 4517.0,
       'batch_size': 727.6171125587168,
       'num_units_layer_1': 4634.0,
       'step': 42.0,
       'std_layer_1': 0.0004299309081302,
       'momentum': 0.6321801704123293,
       'num_epochs': 371.0,
       'std_layer_3': 0.028926029974051224,
       'std_layer_2': 0.01145562004800013,
       'lr_policy': 'inv',
       'learning_rate': 0.029826318066865545,
       'gamma': 0.46436150446432656}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.7238401506457167,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.3233543463764956,
       'dropout_output': 0.387692658865333,
       'learning_rate': 0.25055705607149903,
       'dropout_layer_2': 0.16537451177082238,
       'num_units_layer_2': 5217.0,
       'batch_size': 287.15312680026267,
       'num_units_layer_1': 2623.0,
       'step': 73.0,
       'rho': 0.3993788398015923,
       'num_epochs': 412.0,
       'std_layer_2': 0.005468903582612894,
       'lr_policy': 'inv',
       'std_layer_1': 0.03196564357281583,
       'gamma': 0.7225253602187486}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.9084427363360542,
       'solver': '"sgd"',
       'dropout_layer_1': 0.47408618114026724,
       'dropout_output': 0.6375324638845747,
       'dropout_layer_3': 0.13532008977088042,
       'dropout_layer_2': 0.10867051992136886,
       'num_units_layer_2': 5200.0,
       'num_units_layer_3': 1764.0,
       'batch_size': 751.7233769160604,
       'num_units_layer_1': 1125.0,
       'step': 84.0,
       'std_layer_1': 0.004706707945838345,
       'momentum': 0.6579670949791995,
       'num_epochs': 418.0,
       'std_layer_3': 0.06703203686923141,
       'std_layer_2': 0.021364776685813765,
       'lr_policy': 'exp',
       'learning_rate': 1.006895930668558e-06,
       'gamma': 0.26589296318210975}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.8808203083192006,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.31614542177669236,
       'dropout_output': 0.6128834950904959,
       'dropout_layer_3': 0.6734939376905885,
       'dropout_layer_2': 0.5921620400263008,
       'num_units_layer_2': 300.0,
       'num_units_layer_3': 3672.0,
       'batch_size': 241.71188208498393,
       'num_units_layer_1': 214.0,
       'step': 56.0,
       'std_layer_1': 0.05214945083378918,
       'momentum': 0.6720757416354959,
       'num_epochs': 194.0,
       'std_layer_3': 0.02020291659500061,
       'std_layer_2': 0.08489080217582551,
       'lr_policy': 'exp',
       'learning_rate': 0.28728739883922444,
       'gamma': 0.8087528472911654}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.9982859561242945,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.32925257461740814,
       'dropout_output': 0.1924704598298523,
       'learning_rate': 0.2135562949651281,
       'dropout_layer_2': 0.5346257876302651,
       'num_units_layer_2': 3885.0,
       'batch_size': 793.5704826524803,
       'num_units_layer_1': 4593.0,
       'step': 91.0,
       'momentum': 0.47096093223996227,
       'num_epochs': 488.0,
       'std_layer_2': 0.007225888809513894,
       'lr_policy': 'exp',
       'std_layer_1': 0.0019427866308684189,
       'gamma': 0.34356010533685133}, 0.02),
     ({'num_layers': 3.0,
       'power': 0.951501985775478,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.7836649078292658,
       'dropout_output': 0.2399093307780398,
       'learning_rate': 0.010929876171533532,
       'dropout_layer_2': 0.011586976261760496,
       'num_units_layer_2': 5787.0,
       'batch_size': 31.814264805340258,
       'num_units_layer_1': 4722.0,
       'step': 18.0,
       'momentum': 0.8768999623562101,
       'num_epochs': 486.0,
       'std_layer_2': 0.012236381842172813,
       'lr_policy': 'step',
       'std_layer_1': 0.0024717223220700455,
       'gamma': 0.9667180658562067}, 0.10000000000000001),
     ({'num_layers': 3.0,
       'power': 0.8196900527717096,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.3660063483123479,
       'dropout_output': 0.2553598975807717,
       'learning_rate': 0.0792359335703241,
       'dropout_layer_2': 0.17148792984538072,
       'num_units_layer_2': 4800.0,
       'batch_size': 242.74838348417052,
       'num_units_layer_1': 4749.0,
       'step': 37.0,
       'momentum': 0.7768588564816645,
       'num_epochs': 494.0,
       'std_layer_2': 0.08438075257417156,
       'lr_policy': 'exp',
       'std_layer_1': 0.0015575939075707984,
       'gamma': 0.7240342867167526}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.9982859561242945,
       'solver': '"nesterov"',
       'dropout_layer_1': 0.32925257461740814,
       'dropout_output': 0.1924704598298523,
       'dropout_layer_3': 0.05910805094837215,
       'dropout_layer_2': 0.5346257876302651,
       'num_units_layer_2': 1555.0,
       'num_units_layer_3': 3709.0,
       'batch_size': 793.5704826524803,
       'num_units_layer_1': 4593.0,
       'step': 91.0,
       'std_layer_1': 0.0019427866308684189,
       'momentum': 0.47096093223996227,
       'num_epochs': 488.0,
       'std_layer_3': 0.004455993940858878,
       'std_layer_2': 0.07300638871271027,
       'lr_policy': 'fixed',
       'learning_rate': 0.1122446352398429,
       'gamma': 0.7159714609304522}, 0.02),
     ({'num_layers': 4.0,
       'power': 0.7038076018250163,
       'solver': '"adadelta"',
       'dropout_layer_1': 0.21049439968187753,
       'dropout_output': 0.37339465970155605,
       'dropout_layer_3': 0.4492220124816988,
       'dropout_layer_2': 0.2556332708684139,
       'num_units_layer_2': 2212.0,
       'num_units_layer_3': 5926.0,
       'batch_size': 934.4929652600218,
       'num_units_layer_1': 1350.0,
       'step': 23.0,
       'std_layer_1': 0.00041345564765364986,
       'rho': 0.6263095724704066,
       'num_epochs': 57.0,
       'std_layer_3': 0.058349495588107474,
       'std_layer_2': 0.04610450347271679,
       'lr_policy': 'fixed',
       'learning_rate': 7.219811931684516e-05,
       'gamma': 0.5372817819519713}, 0.02),
     ({'dropout_output': 0.05419711953824402,
       'num_units_layer_4': 503.0,
       'num_epochs': 249.0,
       'num_units_layer_2': 1798.0,
       'num_units_layer_3': 4435.0,
       'num_units_layer_1': 5785.0,
       'num_layers': 5.0,
       'std_layer_3': 0.0020693367447187833,
       'std_layer_2': 0.01626605159391926,
       'std_layer_1': 0.02160469296878519,
       'learning_rate': 0.161203081475512,
       'std_layer_4': 0.04266796418763915,
       'power': 0.7730778282076188,
       'dropout_layer_1': 0.5996870640229259,
       'dropout_layer_3': 0.2387361343390475,
       'dropout_layer_2': 0.049116767959870425,
       'dropout_layer_4': 0.6672322817450616,
       'batch_size': 719.3714049032842,
       'step': 14.0,
       'lr_policy': 'fixed',
       'solver': '"adagrad"',
       'gamma': 0.10402671251350482}, 0.02),
     ({'dropout_output': 0.1570217658484297,
       'num_units_layer_4': 5074.0,
       'num_units_layer_5': 4907.0,
       'num_units_layer_2': 5709.0,
       'num_units_layer_3': 5344.0,
       'num_units_layer_1': 2394.0,
       'num_epochs': 439.0,
       'std_layer_3': 0.07601832353856731,
       'std_layer_2': 0.020438631464011356,
       'lr_policy': 'step',
       'learning_rate': 0.0010928685578229672,
       'momentum': 0.7135023584881737,
       'std_layer_4': 0.05067105296151777,
       'power': 0.7904179532313478,
       'dropout_layer_1': 0.407964865575413,
       'dropout_layer_3': 0.1626118699682814,
       'dropout_layer_2': 0.8417694944058159,
       'dropout_layer_5': 0.5575846370595227,
       'dropout_layer_4': 0.7007827393164742,
       'batch_size': 702.817516948513,
       'step': 3.0,
       'std_layer_1': 0.011027585261841283,
       'num_layers': 6.0,
       'solver': '"nesterov"',
       'std_layer_5': 0.0986334443081331,
       'gamma': 0.41981866741792373}, 0.02)]

targets = []
predictions = []
predictions_valid = []
predictions_test = []

# Make predictions and weight them
iteration = 0
for config, weight in choices:
    print(config)
    print(weight)

    # Extract layer specific variables
    num_layers = int(config["num_layers"])
    num_units_per_layer = []
    dropout_per_layer = []
    std_per_layer = []
    for i in range(1, num_layers):
        num_units_per_layer.append(int(config["num_units_layer_" + str(i)]))
        dropout_per_layer.append(float(config["dropout_layer_" + str(i)]))
        std_per_layer.append(float(config["std_layer_" + str(i)]))

    # Extract solver specific variables
    if config["solver"] == "adam":
        beta1 = config["beta1"]
        beta2 = config["beta2"]
    else:
        beta1 = None
        beta2 = None

    if config["solver"] == "adadelta":
        rho = config["rho"]
    else:
        rho = None

    if config["solver"] == "momentum" or config["solver"] == "nesterov":
        momentum = config["momentum"]
    else:
        momentum = None
    iteration += 1

    n_feat = X_train.shape[1]
    num_output_units = dm.info["label_num"]

    input_shape = (int(config["batch_size"]), 1, 1, n_feat)

    net = FeedForwardNet(input_shape=input_shape,
                     batch_size=config["batch_size"],
                     num_layers=num_layers,
                     num_units_per_layer=num_units_per_layer,
                     dropout_per_layer=dropout_per_layer,
                     std_per_layer=std_per_layer,
                     num_output_units=num_output_units,
                     dropout_output=config["dropout_output"],
                     learning_rate=config["learning_rate"],
                     momentum=momentum,
                     beta1=beta1,
                     beta2=beta2,
                     rho=rho,
                     solver=config["solver"])

    print(X_train[:, np.newaxis, np.newaxis, :].shape)
    validation_error, learning_curve = train(net,
                                             X_train[:, np.newaxis, np.newaxis, :],
                                             Y_train,
                                             X_train[:, np.newaxis, np.newaxis, :],
                                             Y_train,
                                             num_epochs=int(config["num_epochs"]),
                                             lr_policy=config["lr_policy"],
                                             gamma=config["gamma"],
                                             power=config["power"],
                                             step=config["step"])
    print ("Learning Curve")
    print (learning_curve)

    # Compute predctions of this fold
    predictions_valid.append(lasagne.layers.get_output(net.network, X_valid[:, np.newaxis, np.newaxis, :], deterministic=True).eval() * weight)
    predictions_test.append(lasagne.layers.get_output(net.network, X_test[:, np.newaxis, np.newaxis, :], deterministic=True).eval() * weight)


# Output the predictions
for name, predictions in [('valid', predictions_valid),
                          ('test', predictions_test)]:
    predictions = np.array(predictions)
    predictions = np.sum(predictions, axis=0).astype(np.float32)

    filepath = os.path.join(output, '%s_%s_000.predict' % (dataset, name))
    np.savetxt(filepath, predictions, delimiter=' ', fmt='%.4e')
