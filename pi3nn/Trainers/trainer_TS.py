
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math

from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.Networks.networks import CL_UQ_Net_train_steps
from pi3nn.Visualizations.visualization import CL_plotter
from pi3nn.Optimizations.boundary_optimizer import CL_boundary_optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CL_trainer_TS:

    def __init__(self, configs, net_mean, net_std_up, net_std_down, xTrain, yTrain, xValid=None, yValid=None, xTest=None, yTest=None, testDataEvaluationDuringTrain=False):

        self.bool_Nan = False
        self.configs = configs
        self.net_mean = net_mean
        self.net_up = net_up
        self.net_down = net_down

        self.xTrain = xTrain
        self.yTrain = yTrain

        if xValid is not None:
            self.xValid = xValid
        if yValid is not None:
            self.yValid = yValid

        if xTest is not None:
            self.xTest = xTest
        if yTest is not None:
            self.yTest = yTest


