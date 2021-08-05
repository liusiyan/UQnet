import numpy as np
import random

from pathlib import Path
import os

# tf.config.set_visible_devices([], 'GPU')
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.random.set_seed(seed)

''' The OOD related N-dimensional function generator
'''

class CL_function_generator:
    ''' Function and synthetic data generator'''
    def __init__(self, generator_rnd_seed=None):
        self.generator_rnd_seed = generator_rnd_seed
        random.seed(self.generator_rnd_seed)
        np.random.seed(self.generator_rnd_seed)

    def generate_y(self, x_arr, x_dim, coeff_list, power_list,
                          denominator=None,
                          noise=False,
                          noise_type=None,
                          noise_amplitude=None,
                          noise_shift=0):
        ## INPUT: x_arr can be 1-D (m, 1) or n-D (m, n) 2D matrix
        ## OUTPUT: 1-D y matrix (m, 1)

        x = x_arr.astype(np.float32)
        y = np.zeros((x.shape[0], 1))

        # print('x shape:{}'.format(x.shape))
        # print('y shape:{}'.format(y.shape))

        if x_dim == 1:
            # for i in range(y.shape[0]):
            #     y[i, 0] = coeff_list[0] * x[i][0] ** power_list[0]  # suppose coeff_list and power list has only one element
            # vectorize
            y = coeff_list[0] * x ** power_list[0]
        else:  ## multi-dimension
            for i in range(y.shape[0]):
                for j in range(x.shape[1]):
                    y[i, 0] += coeff_list[j] * x[i][j] ** power_list[j]

        if denominator is not None:
            y = y / denominator

        if noise:
            sigma = noise_amplitude * np.ones_like(y) + np.ones_like(y) * noise_shift
            r = np.random.normal(0, sigma).astype(np.float32)
            y = y + r

        return y
