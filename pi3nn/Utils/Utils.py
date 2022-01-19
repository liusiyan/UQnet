import numpy as np



class CL_Utils:

    def __init__(self):
        pass

    def standardizer(self, input_np):
        input_mean = input_np.mean(axis=0, keepdims=1)
        input_std = input_np.std(axis=0, keepdims=1)
        input_std[input_std < 1e-10] = 1.0
        standardized_input = (input_np - input_mean) / input_std
        return standardized_input, input_mean, input_std


    def getNumInputsOutputs(self, inputsOutputs_np):
        if len(inputsOutputs_np.shape) == 1:
            numInputsOutputs = 1
        if len(inputsOutputs_np.shape) > 1:
            numInputsOutputs = inputsOutputs_np.shape[1]
        return numInputsOutputs


''' The early stopping method is taken and adjusted from Keras/Tensorflow EarlyStopping callback function '''
class CL_EarlyStopping:

    def __init__(self,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            print('--- Warning: EarlyStopping mode {} is unknown'.format(mode), 'auto mode will be used')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def init_early_stopping(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            # self.best = np.Inf
        self.best_weights = None


    def earlyStop_onEpochEnd_eval(self, epoch, current_loss):
        if self.monitor_op(current_loss - self.min_delta, self.best):
            self.best = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get


    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

