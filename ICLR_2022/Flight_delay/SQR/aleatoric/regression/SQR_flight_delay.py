# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import six
import sys
sys.modules['sklearn.externals.six'] = six
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import os
import torch
device = 'cpu'
print('device', device)
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from scipy import stats
import pandas as pd


import numpy as np
import statsmodels.api as sm
from skgarden import RandomForestQuantileRegressor
from sklearn import ensemble

import data_loader

from scipy.stats import norm
import matplotlib.pyplot as plt

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class LinearQR:
    """
    Estimate conditional quantiles using linear Quantile Regression
    (fits one model per quantile)
    """
    def __init__(self, x, y, args):
        super(LinearQR, self).__init__()
        self.model = sm.QuantReg(y, x)
        self.alpha = args.alpha
        self.model_name = "LinearQR"

    def predict(self, x_te):
        preds_low = self.model.fit(q = self.alpha / 2).predict(x_te)
        preds_high = self.model.fit(q = (1 - self.alpha / 2)).predict(x_te)
        preds_mean = (preds_high - preds_low) / 2

        return torch.Tensor(preds_mean), torch.Tensor(preds_low), torch.Tensor(preds_high)


class GradientBoostingQR:
    """
    Estimate conditional quantiles by Gradient Boosting
    (fits one model per quantile)
    """
    def __init__(self, x, y, args):
        super(GradientBoostingQR, self).__init__()
        self.alpha = args.alpha
        self.model_name = "GradientBoostingQR"

        self.gbf_low = ensemble.GradientBoostingRegressor(loss='quantile',
                                                     alpha=self.alpha / 2,
                                                     n_estimators=args.n_learners,
                                                     max_depth=args.max_depth,
                                                     learning_rate=args.lr,
                                                     min_samples_leaf=args.min_samples_leaf,
                                                     min_samples_split=args.min_samples_split)

        self.gbf_low.fit(x, y)

        self.gbf_high = ensemble.GradientBoostingRegressor(loss='quantile',
                                                           alpha=(1 - self.alpha / 2),
                                                           n_estimators=args.n_learners,
                                                           max_depth=args.max_depth,
                                                           learning_rate=args.lr,
                                                           min_samples_leaf=args.min_samples_leaf,
                                                           min_samples_split=args.min_samples_split)

        self.gbf_high.fit(x, y)

    def predict(self, x_te):
        preds_low = self.gbf_low.predict(x_te)
        preds_high = self.gbf_high.predict(x_te)
        preds_mean = (preds_high - preds_low) / 2

        return torch.Tensor(preds_mean), torch.Tensor(preds_low), torch.Tensor(preds_high)

class RandomForestQR:
    """
    Estimate conditional quantiles by Random Forests
    (fits one model for all quantiles)
    """
    def __init__(self, x, y, args):
        super(RandomForestQR, self).__init__()
        self.alpha = args.alpha
        self.model_name = "RandomForestQR"
        self.rf = ensemble.RandomForestRegressor(n_estimators=args.n_learners,
                                                 min_samples_leaf=args.min_samples_leaf,
                                                 random_state=args.seed,
                                                 verbose=False,
                                                 n_jobs=-1)
        self.rf.fit(x, y)

    def predict(self, x_te):
        rf_preds = []
        for estimator in self.rf.estimators_:
            rf_preds.append(estimator.predict(x_te))
        rf_preds = np.array(rf_preds).transpose()

        preds_low = np.percentile(rf_preds, (self.alpha / 2) * 100, axis=1)
        preds_high = np.percentile(rf_preds, (1 - self.alpha / 2) * 100, axis=1)
        preds_mean = (preds_high - preds_low) / 2

        return torch.Tensor(preds_mean), torch.Tensor(preds_low), torch.Tensor(preds_high)


class QuantileForest:
    """
    Estimate conditional quantiles by Quantile Forest
    (fits one model for all quantiles)
    """
    def __init__(self, x, y, args):
        super(QuantileForest, self).__init__()
        self.alpha = args.alpha
        self.model_name = "QuantileForest"
        self.rfqr = RandomForestQuantileRegressor(n_estimators=args.n_learners)
                                                  #min_samples_split=args.min_samples_split,
                                                  #n_estimators=args.n_learners,
                                                  #random_state=args.seed)
        # self.rfqr.set_params(max_features=x.shape[1] // args.max_features)
        self.rfqr.fit(x, y)

    def predict(self, x_te):
        preds_low = self.rfqr.predict(x_te, (self.alpha / 2) * 100)
        preds_high = self.rfqr.predict(x_te, (1 - self.alpha / 2) * 100)
        preds_mean = (preds_high - preds_low) / 2

        return torch.Tensor(preds_mean), torch.Tensor(preds_low), torch.Tensor(preds_high)


class QuantileLoss(torch.nn.Module):
    """
    Quantile regression loss
    """

    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, yhat, y, tau):
        diff = yhat - y
        mask = (diff.ge(0).float() - tau).detach()
        return (mask * diff).mean()


class NegativeLogLikelihoodLoss(torch.nn.Module):
    """
    Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
    Equation (1)
    (https://arxiv.org/abs/1612.01474)
    """

    def __init__(self):
        super(NegativeLogLikelihoodLoss, self).__init__()

    def forward(self, yhat, y):
        mean = yhat[:, 0].view(-1, 1)
        variance = yhat[:, 1].view(-1, 1)

        # make variance positive and stable (footnote 2)
        variance2 = variance.exp().add(1).log().add(0.001)

        return (variance2.log().div(2) + (y - mean).pow(2).div(variance2.mul(2))).mean()


class QualityDrivenLoss(torch.nn.Module):
    """
    High-Quality Prediction Intervals for Deep Learning
    Equation (15)
    (https://arxiv.org/pdf/1802.07167.pdf)
    """

    def __init__(self,
                 alpha=0.05,
                 lamba=15.,
                 soften=160):
        super(QualityDrivenLoss, self).__init__()

        self.alpha = alpha
        self.lamba = lamba
        self.soften = soften
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, yhat, y):
        y_l = yhat[:, 0].view(-1, 1)
        y_u = yhat[:, 1].view(-1, 1)

        k_u_soft = self.sigmoid(self.soften * (y_u - y))
        k_l_soft = self.sigmoid(self.soften * (y - y_l))
        k_soft = k_u_soft * k_l_soft
        picp_soft = k_soft.mean()
        mpiw_soft = ((y_u - y_l).abs() * k_soft).sum() / \
            k_soft.sum().add(0.001)

        k_u_hard = (y_u - y).sign().clamp(0)
        k_l_hard = (y - y_l).sign().clamp(0)
        k_hard = k_u_hard * k_l_hard
        picp_hard = k_hard.mean()
        mpiw_hard = ((y_u - y_l).abs() * k_hard).sum() / \
            k_hard.sum().add(0.001)

        return mpiw_hard + self.lamba * ((1 - self.alpha) - picp_soft).clamp(0).pow(2)


class Perceptron(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_layers,
                 n_hiddens,
                 alpha,
                 dropout):
        super(Perceptron, self).__init__()

        layers = []

        if n_layers == 0:
            layers.append(torch.nn.Linear(n_inputs, n_outputs))
        else:
            layers.append(torch.nn.Linear(n_inputs, n_hiddens))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))

            for layer in range(n_layers - 1):
                layers.append(torch.nn.Linear(n_hiddens, n_hiddens))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))

            layers.append(torch.nn.Linear(n_hiddens, n_outputs))

        self.perceptron = torch.nn.Sequential(*layers)
        self.loss_function = None

    def loss(self, x, y):
        return self.loss_function(self.perceptron(x), y)


class Ensemble(torch.nn.Module):
    def __init__(self,
                 network_name,
                 n_ens,
                 n_inputs,
                 n_outputs,
                 n_layers,
                 n_hiddens,
                 alpha,
                 dropout):
        super(Ensemble, self).__init__()

        # choose network
        extra_inputs = 0
        extra_outputs = 0
        effective_dropout = 0

        if network_name == "QualityDriven":
            BaseModel = QualityDriven
            extra_outputs = 1
        elif network_name == "ConditionalGaussian":
            BaseModel = ConditionalGaussian
            extra_outputs = 1
        elif network_name == "ConditionalQuantile":
            BaseModel = ConditionalQuantile
            extra_inputs = 1
        elif network_name == "Dropout":
            BaseModel = Dropout
            effective_dropout = dropout

        self.alpha = alpha
        self.learners = torch.nn.ModuleList()

        for _ in range(n_ens):
            self.learners.append(BaseModel(n_inputs=n_inputs + extra_inputs,
                                           n_outputs=n_outputs + extra_outputs,
                                           n_layers=n_layers,
                                           n_hiddens=n_hiddens,
                                           alpha=alpha,
                                           dropout=effective_dropout))

    def predict(self, x):
        preds_mean = torch.zeros(len(self.learners), x.size(0), 1)
        preds_low = torch.zeros(len(self.learners), x.size(0), 1)
        preds_high = torch.zeros(len(self.learners), x.size(0), 1)

        for l, learner in enumerate(self.learners):
            preds_mean[l], preds_low[l], preds_high[l] = learner.predict(x)

        m = len(self.learners)
        
        threshold = norm.ppf(self.alpha / 2)

        preds_mean = preds_mean.mean(0)
        preds_low = preds_low.mean(0) - threshold * preds_low.std(0, m > 1)
        preds_high = preds_high.mean(0) + threshold * preds_high.std(0, m > 1)

        return preds_mean, preds_low, preds_high

    def loss(self, x, y):
        loss = 0
        for learner in self.learners:
            loss += learner.loss(x, y)

        return loss


class Dropout(Perceptron):
    def __init__(self, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.loss_function = torch.nn.MSELoss()
        self.t = kwargs["alpha"] * 100 / 2

    def predict(self, x, reps=1000):
        preds = torch.zeros(x.size(0), reps)

        for rep in range(reps):
            preds[:, rep] = self.perceptron(x)[:, 0].detach()

        pred_low = torch.Tensor(np.percentile(preds.numpy(), self.t, 1))
        pred_high = torch.Tensor(np.percentile(preds.numpy(), 100 - self.t, 1))

        return preds.mean(1).view(-1, 1), pred_low.view(-1, 1), pred_high.view(-1, 1)


class QualityDriven(Perceptron):
    def __init__(self, **kwargs):
        super(QualityDriven, self).__init__(**kwargs)
        self.loss_function = QualityDrivenLoss(alpha=kwargs["alpha"])
        self.alpha = kwargs["alpha"]

    def predict(self, x):
        predictions = self.perceptron(x).detach()
        low = predictions[:, 0].view(-1, 1)
        high = predictions[:, 1].view(-1, 1)

        return (low + high) / 2.0, low, high


class ConditionalGaussian(Perceptron):
    def __init__(self, **kwargs):
        super(ConditionalGaussian, self).__init__(**kwargs)
        self.loss_function = NegativeLogLikelihoodLoss()
        self.alpha = kwargs["alpha"] 

    def predict(self, x):
        predictions = self.perceptron(x).detach()
        mean = predictions[:, 0].view(-1, 1)
        var = predictions[:, 1].view(-1, 1)
        var = var.exp().add(1).log().add(1e-6)
        interval = var.sqrt().mul(norm.ppf(self.alpha / 2))

        return mean, mean - interval, mean + interval


class ConditionalQuantile(Perceptron):
    def __init__(self, **kwargs):
        super(ConditionalQuantile, self).__init__(**kwargs)
        self.loss_function = QuantileLoss()
        self.alpha = kwargs["alpha"]

    def predict(self, x):
        tau_l = torch.zeros(x.size(0), 1) + self.alpha / 2
        tau_u = torch.zeros(x.size(0), 1) + (1 - self.alpha / 2) 

        preds_l = self.perceptron(
            torch.cat((x, (tau_l - 0.5) * 12), 1)).detach()
        preds_u = self.perceptron(
            torch.cat((x, (tau_u - 0.5) * 12), 1)).detach()

        return (preds_l + preds_u) / 2, preds_l, preds_u

    def loss(self, x, y):
        tau_l = torch.zeros(x.size(0), 1) + self.alpha / 2
        tau_u = torch.zeros(x.size(0), 1) + (1 - self.alpha / 2) 

        preds_l = self.perceptron(torch.cat((x, (tau_l - 0.5) * 12), 1))
        preds_u = self.perceptron(torch.cat((x, (tau_u - 0.5) * 12), 1))

        return self.loss_function(preds_l, y, tau_l) + self.loss_function(preds_u, y, tau_u)


class CL_cubic_10D:
    def __init__(self):
        pass
    def target_fun(self, x):
        y = x**3
        y = torch.sum(y, dim=1, keepdim=True)/10.0 + 5.0*torch.randn(x.size(0), device=device).unsqueeze(1)
        return y      



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="flight_delay")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--n_hidden_layers', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--n_hidden_units', type=int, default=64)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--ens', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--n_ens', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_learners', type=int, default=1000)
    parser.add_argument('--min_samples_leaf', type=int, default=9)
    parser.add_argument('--min_samples_split', type=int, default=9)
    parser.add_argument('--max_depth', type=int, default=9)
    parser.add_argument('--max_features', type=int, default=3)
    args = parser.parse_args()

    reset_seeds(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # cuda0 = torch.device('cuda:0')


    ##### test Pre-split UCI datasets on on the SingleModelUncertainty method 
    configs = {'data_name':'flight_delay',  
               'original_data_path': '../../../flight_delay_data/',               
               'seed': 12345} 

    configs['data_name'] = args.dataset

    if configs['data_name'] == 'flight_delay':

        SQR_results_path = './SQR_flight_delay_results/'
        ### load flight delay data 
        xTrain, yTrain, test_data_list = data_loader.load_flight_delays(configs['original_data_path'])


    ########  parameters
    test_idx = 0
     # [0, 1, 2, 3] for test 1,2,3,4

    bool_valid = True
    valid_fraction = 0.2

    args.n_epochs = 500
    args.alpha = 0.05

    seed = 12345
    neurons = 100

    # num_neurons_list = [50, 75, 100]
    # lr_list = [1e-2, 1e-3, 1e-4]
    # dropout_list = [0.1, 0.25, 0.5, 0.75]
    # wd_list = [0, 1e-3, 1e-2, 1e-1, 1]

    args.n_hidden_units = neurons
    args.lr = 1e-3
    args.dropout = 0.1
    args.wd = 0

    dataName = args.dataset

    xTest = test_data_list[test_idx][0]
    yTest = test_data_list[test_idx][1]

    y_al = np.concatenate((yTrain, yTest))
    y_al = torch.Tensor(y_al)


    if bool_valid:
        xTrain, xValid, yTrain, yValid = train_test_split(xTrain, yTrain, test_size=valid_fraction, random_state=args.seed)

    reset_seeds(args.seed)

    x_tr = torch.Tensor(xTrain)
    x_va = torch.Tensor(xValid)
    x_te = torch.Tensor(xTest)

    y_tr = torch.Tensor(yTrain.reshape(-1, 1))
    y_va = torch.Tensor(yValid.reshape(-1, 1))
    y_te = torch.Tensor(yTest.reshape(-1, 1))
    y_al = torch.Tensor(y_al.reshape(-1, 1))

    for network_name in ["ConditionalQuantile"]:
        reset_seeds(args.seed)
        # print(x_tr.shape)
        # print(y_tr.shape)
        network = Ensemble(network_name, args.n_ens,
                           x_tr.size(1), y_tr.size(1),
                           args.n_hidden_layers, args.n_hidden_units,
                           args.alpha, args.dropout)

        loader_tr = DataLoader(TensorDataset(x_tr, y_tr),
                               shuffle=True,
                               batch_size=args.bs)

        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.wd)

        for epoch in range(args.n_epochs):
            if epoch%100 == 0:
                print('--- Epoch: {}'.format(epoch))
            for (xi, yi) in loader_tr:
                # xi, yi = xi.cuda(), yi.cuda()
                optimizer.zero_grad()
                network.loss(xi, yi).backward()
                optimizer.step()

        # make predictions
        p_mean_tr, p_low_tr, p_high_tr = network.predict(x_tr)
        # p_mean_va, p_low_va, p_high_va = network.predict(x_va)
        p_mean_te, p_low_te, p_high_te = network.predict(x_te)

        # final losses
        mse_tr = network.loss(x_tr, y_tr)
        # mse_va = network.loss(x_va, y_va)
        mse_te = network.loss(x_te, y_te)

        # percentage of captured points
        capture_tr = (p_low_tr.lt(y_tr) * y_tr.lt(p_high_tr)).float().mean()
        # capture_va = (p_low_va.lt(y_va) * y_va.lt(p_high_va)).float().mean()
        capture_te = (p_low_te.lt(y_te) * y_te.lt(p_high_te)).float().mean()

        # width of intervals
        y_range = (y_al.max() - y_al.min())
        width_tr = (p_high_tr - p_low_tr).abs().mean() / y_range
        # width_va = (p_high_va - p_low_va).abs().mean() / y_range
        width_te = (p_high_te - p_low_te).abs().mean() / y_range

        if bool_valid:
            p_mean_va, p_low_va, p_high_va = network.predict(x_va)
            mse_va = network.loss(x_va, y_va)
            capture_va = (p_low_va.lt(y_va) * y_va.lt(p_high_va)).float().mean()
            width_va = (p_high_va - p_low_va).abs().mean() / y_range


        if bool_valid:
            print("{:<22} | {:<26} | {:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} | {:<2} | {:<4} | {} | {}".format(
            network_name + "-" + str(args.n_ens), args.dataset,
            mse_tr, capture_tr, width_tr,
            mse_va, capture_va, width_va,
            mse_te, capture_te, width_te,
            args.seed,
            epoch,
            args.lr,
            args.wd))
        else:
            print("{:<22} | {:<26} | {:.5f} {:.5f} {:.5f} | {:.5f} {:.5f} {:.5f} | {:<2} | {:<4} | {} | {}".format(
                network_name + "-" + str(args.n_ens), args.dataset,
                mse_tr, capture_tr, width_tr,
                # mse_va, capture_va, width_va,
                mse_te, capture_te, width_te,
                args.seed,
                epoch,
                args.lr,
                args.wd))

        mse_tr, capture_tr, width_tr = mse_tr.detach().numpy(), capture_tr.detach().numpy(), width_tr.detach().numpy()
        # mse_va, capture_va, width_va = mse_va.detach().numpy(), capture_va.detach().numpy(), width_va.detach().numpy()
        mse_te, capture_te, width_te = mse_te.detach().numpy(), capture_te.detach().numpy(), width_te.detach().numpy()

        if bool_valid:
            mse_va, capture_va, width_va = mse_va.detach().numpy(), capture_va.detach().numpy(), width_va.detach().numpy()


        ### calculate the confidence scores
        MPIW_array_train = (p_high_tr - p_low_tr).detach().numpy().flatten()
        MPIW_array_test = (p_high_te - p_low_te).detach().numpy().flatten()
        MPIW_train = np.mean(MPIW_array_train)

        confidence_arr_test = [min(MPIW_train/test_width, 1.0) for test_width in MPIW_array_test]
        confidence_arr_train = [min(MPIW_train/train_width, 1.0) for train_width in MPIW_array_train]

        print('----------- OOD analysis --- confidence scores ----------------')
        print('--- Train conf_scores MEAN: {}, STD: {}'.format(np.mean(confidence_arr_train), np.std(confidence_arr_train)))
        print('--- Test: {} rank: {} conf_scores MEAN: {}, STD: {}'.format(test_idx+1, test_idx+1, np.mean(confidence_arr_test), np.std(confidence_arr_test)))


