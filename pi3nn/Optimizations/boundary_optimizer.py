''' Upper and lower boundary optimization '''

import numpy as np

class CL_boundary_optimizer:
    def __init__(self, yTrain, output_mean, output_up, output_down, num_outlier=None,
                 c_up0_ini=None,
                 c_up1_ini=None,
                 c_down0_ini=None,
                 c_down1_ini=None,
                 max_iter=None):

        self.yTrain = yTrain.flatten()

        self.output_mean = output_mean
        self.output_up = output_up
        self.output_down = output_down
        if num_outlier is not None:
            self.num_outlier = num_outlier
        self.c_up0_ini = c_up0_ini
        self.c_up1_ini = c_up1_ini
        self.c_down0_ini = c_down0_ini
        self.c_down1_ini = c_down1_ini
        self.max_iter = max_iter

    def optimize_up(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_up0 = self.c_up0_ini
        c_up1 = self.c_up1_ini
        f0 = np.count_nonzero(self.yTrain >= self.output_mean.numpy().flatten() + c_up0 * self.output_up.numpy().flatten()) - self.num_outlier
        f1 = np.count_nonzero(self.yTrain >= self.output_mean.numpy().flatten() + c_up1 * self.output_up.numpy().flatten()) - self.num_outlier

        iter = 0
        while iter <= self.max_iter and f0 != 0 and f1 != 0:
            c_up2 = (c_up0 + c_up1) / 2.0
            f2 = np.count_nonzero(
                self.yTrain >= self.output_mean.numpy().flatten() + c_up2 * self.output_up.numpy().flatten()) - self.num_outlier
            if f2 == 0:
                break
            elif f2 > 0:
                c_up0 = c_up2
                f0 = f2
            else:
                c_up1 = c_up2
                f1 = f2
            iter += 1
            if verbose > 1:
                print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
                print('c_up0: {}, c_up1: {}, c_up2: {}'.format(c_up0, c_up1, c_up2))
        if verbose > 0:
            print('f0 : {}'.format(f0))
            print('f1 : {}'.format(f1))

        c_up = c_up2
        return c_up

    def optimize_down(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_down0 = self.c_down0_ini
        c_down1 = self.c_down1_ini
        f0 = np.count_nonzero(self.yTrain <= self.output_mean.numpy().flatten() - c_down0 * self.output_down.numpy().flatten()) - self.num_outlier
        f1 = np.count_nonzero(self.yTrain <= self.output_mean.numpy().flatten() - c_down1 * self.output_down.numpy().flatten()) - self.num_outlier

        iter = 0
        while iter <= self.max_iter and f0 != 0 and f1 != 0:
            c_down2 = (c_down0 + c_down1) / 2.0
            f2 = np.count_nonzero(self.yTrain <= self.output_mean.numpy().flatten() - c_down2 * self.output_down.numpy().flatten()) - self.num_outlier
            if f2 == 0:
                break
            elif f2 > 0:
                c_down0 = c_down2
                f0 = f2
            else:
                c_down1 = c_down2
                f1 = f2
            iter += 1
            if verbose > 1:
                print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
                print('c_down0: {}, c_down1: {}, c_down2: {}'.format(c_down0, c_down1, c_down2))
        if verbose > 0:
            print('f0 : {}'.format(f0))
            print('f1 : {}'.format(f1))

        c_down = c_down2
        return c_down


class CL_prediction_shifter:
    def __init__(self, configs, yTrain, output_mean, num_outlier=None,
                 c_m0_ini=None,
                 c_m1_ini=None,
                 max_iter=None):

        self.configs = configs
        self.yTrain = yTrain
        self.Ntrain = self.yTrain.shape[0]
        self.output_mean = output_mean

        self.c_m0 = c_m0_ini
        self.c_m1 = c_m1_ini
        self.max_iter = max_iter

    def median_shift(self, verbose=0):
        self.yTrain = self.yTrain.flatten()
        half_num_data = int(self.Ntrain / 2)
        f0 = np.count_nonzero(self.yTrain >= self.output_mean.numpy().flatten() + self.c_m0 * self.output_mean.numpy().flatten()) - half_num_data
        f1 = np.count_nonzero(self.yTrain >= self.output_mean.numpy().flatten() + self.c_m1 * self.output_mean.numpy().flatten()) - half_num_data
        iter = 0
        while iter <= self.max_iter and f0 != 0 and f1 != 0:
            self.c_m2 = (self.c_m0 + self.c_m1) / 2.0
            f2 = np.count_nonzero(
                self.yTrain >= self.output_mean.numpy().flatten() + self.c_m2 * self.output_mean.numpy().flatten()) - half_num_data
            if f2 == 0:
                break
            elif f2 > 0:
                self.c_m0 = self.c_m2
                f0 = f2
            else:
                self.c_m1 = self.c_m2
                f1 = f2
            iter += 1
            if verbose > 1:
                print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
                print('c_m0: {}, c_m1: {}, c_m2: {}'.format(self.c_m0, self.c_m1, self.c_m2))
        if verbose > 0:
            print('f0 : {}'.format(f0))
            print('f1 : {}'.format(f1))
            print('c_m0: {}, c_m1: {}, c_m2: {}'.format(self.c_m0, self.c_m1, self.c_m2))
        c_m = self.c_m2
        return c_m
        




