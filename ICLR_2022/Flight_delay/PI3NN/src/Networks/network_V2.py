''' Network structure and training steps '''

import tensorflow as tf
# Force using CPU globally by hiding GPU(s)
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers

''' Network definition '''
class UQ_Net_mean_TF2(Model):
    def __init__(self, num_inputs, num_outputs, num_neurons=None):
        super(UQ_Net_mean_TF2, self).__init__()
        self.inputLayer = Dense(num_inputs, activation='linear')
        initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)
        self.fc1 = Dense(num_neurons, activation='relu',
                         kernel_initializer=initializer,
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
                         )
        # self.fc2 = Dense(50, activation='relu',
        #          kernel_initializer=initializer,
        #          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
        #          )
        # self.fc3 = Dense(10, activation='relu',
        #          kernel_initializer=initializer,
        #          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
        #          )


        self.outputLayer = Dense(num_outputs)

    def call(self, x):
        x = self.inputLayer(x)
        x = self.fc1(x)

        # x = self.fc2(x)
        # x = self.fc3(x)

        x = self.outputLayer(x)
        return x

class UQ_Net_std_TF2(Model):
    def __init__(self,  num_inputs, num_outputs, num_neurons=None, bias=None):
        super(UQ_Net_std_TF2, self).__init__()
        self.inputLayer = Dense(num_inputs, activation='linear')
        initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)
        self.fc1 = Dense(num_neurons, activation='relu',
                         kernel_initializer=initializer,
                         # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
                        )
        # self.fc2 = Dense(50, activation='relu',
        #          kernel_initializer=initializer,
        #          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
        #          )
        # self.fc3 = Dense(10, activation='relu',
        #          kernel_initializer=initializer,
        #          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
        #          )
        self.outputLayer = Dense(num_outputs)
        if bias is None:
            self.custom_bias = tf.Variable([3.0])
        else:
            self.custom_bias = tf.Variable([bias])

    def call(self, x):
        x = self.inputLayer(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.outputLayer(x)
        x = tf.nn.bias_add(x, self.custom_bias)
        x = tf.math.sqrt(tf.math.square(x) + 1e-10)
        return x

class CL_UQ_Net_train_steps:
    # def __init__(self, net_mean,     optimizer_net_mean,
    #                    net_std_up,   optimizer_net_std_up,
    #                    net_std_down, optimizer_net_std_down):
    def __init__(self, net_mean, net_std_up, net_std_down,
                xTrain, yTrain, xTest=None, yTest=None,
                optimizers=['Adam', 'Adam', 'Adam'],
                lr=[0.01, 0.01, 0.01],
                exponential_decay=False,
                decay_steps=None,
                decay_rate=None
        ):

        self.xTrain = xTrain
        self.yTrain = yTrain
        if xTest is not None:
            self.xTest = xTest
        if yTest is not None:
            self.yTest = yTest
            if len(self.yTest.shape) == 1:  # convert to shape (x, 1) from (x,)
                self.yTest = self.yTest.reshape(-1, 1)

        if len(self.yTrain.shape) == 1:  # convert to shape (x, 1) from (x,)
            self.yTrain = self.yTrain.reshape(-1, 1)



        self.exponential_decay = exponential_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.criterion_mean = tf.keras.losses.MeanSquaredError()
        self.criterion_std = tf.keras.losses.MeanSquaredError()

        self.train_loss_net_mean = tf.keras.metrics.Mean(name='train_loss_net_mean')
        self.train_loss_net_std_up = tf.keras.metrics.Mean(name='train_loss_net_std_up')
        self.train_loss_net_std_down = tf.keras.metrics.Mean(name='train_loss_net_std_down')
        # self.valid_loss_net_mean = tf.keras.metrics.Mean(name='valid_loss_net_mean')
        # self.valid_loss_net_std_up = tf.keras.metrics.Mean(name='valid_loss_net_std_up')
        # self.valid_loss_net_std_down = tf.keras.metrics.Mean(name='valid_loss_net_std_down')
        self.test_loss_net_mean = tf.keras.metrics.Mean(name='test_loss_net_mean')
        self.test_loss_net_std_up = tf.keras.metrics.Mean(name='test_loss_net_std_up')
        self.test_loss_net_std_down = tf.keras.metrics.Mean(name='test_loss_net_std_down')

        self.net_mean = net_mean
        if exponential_decay is False:
            if optimizers[0] == 'Adam':
                self.optimizer_net_mean = tf.keras.optimizers.Adam(learning_rate=lr[0])
            elif optimizers[0] == 'SGD':
                self.optimizer_net_mean = tf.keras.optimizers.SGD(learning_rate=lr[0])
        else:
            self.global_step_0 = tf.Variable(0, trainable=False)
            decayed_l_rate_0 = tf.compat.v1.train.exponential_decay(lr[0], self.global_step_0, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
            if optimizers[0] == 'Adam':
                self.optimizer_net_mean = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_0)
            elif optimizers[0] == 'SGD':
                self.optimizer_net_mean = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_0)


        self.net_std_up = net_std_up
        if exponential_decay is False:
            if optimizers[1] == 'Adam':
                self.optimizer_net_std_up = tf.keras.optimizers.Adam(learning_rate=lr[1])
            elif optimizers[1] == 'SGD':
                self.optimizer_net_std_up = tf.keras.optimizers.SGD(learning_rate=lr[1])
        else:
            self.global_step_1 = tf.Variable(0, trainable=False)
            decayed_l_rate_1 = tf.compat.v1.train.exponential_decay(lr[1], self.global_step_1, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
            if optimizers[1] == 'Adam':
                self.optimizer_net_std_up = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_1)
            elif optimizers[1] == 'SGD':
                self.optimizer_net_std_up = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_1)

        self.net_std_down = net_std_down
        if exponential_decay is False:
            if optimizers[2] == 'Adam':
                self.optimizer_net_std_down = tf.keras.optimizers.Adam(learning_rate=lr[2])
            elif optimizers[2] == 'SGD':
                self.optimizer_net_std_down = tf.keras.optimizers.SGD(learning_rate=lr[2])
        else:
            self.global_step_2 = tf.Variable(0, trainable=False)
            decayed_l_rate_2 = tf.compat.v1.train.exponential_decay(lr[2], self.global_step_2, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
            if optimizers[1] == 'Adam':
                self.optimizer_net_std_down = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_2)
            elif optimizers[1] == 'SGD':
                self.optimizer_net_std_down = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_2)


    def add_model_regularizer_loss(self, model):
        loss = 0
        for l in model.layers:
            # if hasattr(l, 'layers') and l.layers:  # the layer itself is a model
            #     loss += add_model_loss(l)
            if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer:
                loss += l.kernel_regularizer(l.kernel)
            if hasattr(l, 'bias_regularizer') and l.bias_regularizer:
                loss += l.bias_regularizer(l.bias)
        return loss

    ''' Training for mean values'''

    @tf.function
    def train_step_UQ_Net_mean_TF2(self, testDataEvaluation=True):
        with tf.GradientTape() as tape:
            train_predictions = self.net_mean(self.xTrain, training=True)
            # print('EEEEEEEEEEEEEEEEEe')
            # tf.print(train_predictions)
            # print(self.yTrain)
            train_loss = self.criterion_mean(self.yTrain, train_predictions)
            ''' Add regularization losses '''
            train_loss += self.add_model_regularizer_loss(self.net_mean)
            if testDataEvaluation == True:
                test_predictions = self.net_mean(self.xTest, training=False)
                test_loss = self.criterion_mean(self.yTest, test_predictions)
            else:
                test_loss = 0
        gradients = tape.gradient(train_loss, self.net_mean.trainable_variables)
        self.optimizer_net_mean.apply_gradients(zip(gradients, self.net_mean.trainable_variables))
        self.train_loss_net_mean(train_loss)
        self.test_loss_net_mean(test_loss)

        if self.exponential_decay:
            self.global_step_0.assign_add(1)

    ''' Training for upper boundary'''
    @tf.function
    def train_step_UQ_Net_std_UP_TF2(self, testDataEvaluation=True):
        with tf.GradientTape() as tape:
            train_predictions = self.net_std_up(self.xTrain, training=True)
            train_loss = self.criterion_std(self.yTrain, train_predictions)
            ''' Add regularization losses '''
            train_loss += self.add_model_regularizer_loss(self.net_std_up)
            if testDataEvaluation == True:
                test_predictions = self.net_std_up(self.xTest, training=False)
                test_loss = self.criterion_std(self.yTest, test_predictions)
            else:
                test_loss = 0
        gradients = tape.gradient(train_loss, self.net_std_up.trainable_variables)
        self.optimizer_net_std_up.apply_gradients(zip(gradients, self.net_std_up.trainable_variables))
        self.train_loss_net_std_up(train_loss)
        self.test_loss_net_std_up(test_loss)

        if self.exponential_decay:
            self.global_step_1.assign_add(1)

    ''' Training for lower boundary '''
    @tf.function
    def train_step_UQ_Net_std_DOWN_TF2(self, testDataEvaluation=True):
        with tf.GradientTape() as tape:
            train_predictions = self.net_std_down(self.xTrain, training=True)
            train_loss = self.criterion_std(self.yTrain, train_predictions)
            ''' Add regularization losses '''
            train_loss += self.add_model_regularizer_loss(self.net_std_down)
            if testDataEvaluation == True:
                test_predictions = self.net_std_down(self.xTest, training=False)
                test_loss = self.criterion_std(self.yTest, test_predictions)
            else:
                test_loss = 0
        gradients = tape.gradient(train_loss, self.net_std_down.trainable_variables)
        self.optimizer_net_std_down.apply_gradients(zip(gradients, self.net_std_down.trainable_variables))
        self.train_loss_net_std_down(train_loss)
        self.test_loss_net_std_down(test_loss)

        if self.exponential_decay:
            self.global_step_2.assign_add(1)
