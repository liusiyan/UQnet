import numpy as np
import tensorflow as tf
import time
import datetime
import os
import sys
import h5py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import evidential_deep_learning as edl
from .util import normalize, gallery

class Ensemble:
    def __init__(self, models, opts, dataset="", learning_rate=1e-3, tag="", custom_plot_folder="", custom_best_results_dat_folder=""):
        self.mse = not opts['sigma']
        self.loss_function = edl.losses.MSE if self.mse else edl.losses.Gaussian_NLL

        self.models = models

        self.num_ensembles = opts['num_ensembles']

        self.optimizers = [tf.optimizers.Adam(learning_rate) for _ in range(self.num_ensembles)]

        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.min_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save','{}_{}_{}_{}'.format(current_time, dataset, trainer, tag))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.custom_plot_folder = custom_plot_folder
        Path(self.custom_plot_folder).mkdir(parents=True, exist_ok=True)

        self.custom_best_results_dat_folder = custom_best_results_dat_folder
        Path(self.custom_best_results_dat_folder).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join('logs', '{}_{}_{}_{}_train'.format(current_time, dataset, trainer, tag))
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs', '{}_{}_{}_{}_val'.format(current_time, dataset, trainer, tag))
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    @tf.function
    def run_train_step(self, x, y):
        losses = []
        y_hats = []
        for (model, optimizer) in zip(self.models, self.optimizers): #Autograph unrolls this so make sure ensemble size is not too large

            with tf.GradientTape() as tape:
                outputs = model(x, training=True) #forward pass
                if self.mse:
                    mu = outputs
                    loss = self.loss_function(y, mu)
                else:
                    mu, sigma = tf.split(outputs, 2, axis=-1)
                    loss = self.loss_function(y, mu, sigma)
                y_hats.append(mu)
                losses.append(loss)

            grads = tape.gradient(loss, model.variables) #compute gradient
            optimizer.apply_gradients(zip(grads, model.variables))

        return tf.reduce_mean(losses), tf.reduce_mean(y_hats, 0)

    @tf.function
    def evaluate(self, x, y):
        preds = tf.stack([model(x, training=False) for model in self.models], axis=0) #forward pass
        if self.mse:
            mean_mu, var = tf.nn.moments(preds, 0)
            loss = self.loss_function(y, mean_mu)
        else:
            mus, sigmas = tf.split(preds, 2, axis=-1)
            mean_mu = tf.reduce_mean(mus, axis=0)
            mean_sigma = tf.reduce_mean(sigmas, axis=0)
            var = tf.reduce_mean(sigmas**2 + tf.square(mus), axis=0) - tf.square(mean_mu)
            loss = self.loss_function(y, mean_mu, tf.sqrt(var))

        rmse = edl.losses.RMSE(y, mean_mu)
        nll = edl.losses.Gaussian_NLL(y, mean_mu, tf.sqrt(var))

        return mean_mu, var, loss, rmse, nll

    def save_train_summary(self, loss, x, y, y_hat):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, y_hat)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(y_hat,idx).numpy())], max_outputs=1, step=self.iter)

    def save_val_summary(self, loss, x, y, mu, var):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, mu)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(mu,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_var", [gallery(normalize(tf.gather(var,idx)).numpy())], max_outputs=1, step=self.iter)

    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        if isinstance(x, tf.Tensor):
            x_ = x[idx,...]
            y_ = y[idx,...]
        elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            idx = np.sort(idx)
            x_ = x[idx,...]
            y_ = y[idx,...]

            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0

            x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
            y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
        else:
            print("unknown dataset type {} {}".format(type(x), type(y)))
        return x_, y_

    def save(self, name):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.save_dir, "{}_{}.h5".format(name, i)))

    ''' Siyan added for testing plot '''
    def plot_scatter_with_var_from_pandas(self, x_train, y_train, x_test, y_test, mu, var, path, n_stds=3, test_bounds=[[-7, +7]], show=True):
        plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label='Train (x_train vs y_train)')
        for k in np.linspace(0, n_stds, 4):
            if k == 0:
                plt.fill_between(x_test[:, 0], (mu - k * var), (mu + k * var), alpha=0.3, edgecolor=None,
                                 facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1, label='Unc.')
            else:
                plt.fill_between(x_test[:, 0], (mu - k * var), (mu + k * var), alpha=0.3, edgecolor=None,
                                 facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)

        plt.plot(x_test, y_test, 'r--', zorder=2, label='True (x_test vs y_test)')
        plt.plot(x_test, mu, color='#007cab', zorder=3, label='Pred (x_test vs mu)')
        plt.gca().set_xlim(*test_bounds)
        plt.gca().set_ylim(-150, 150)
        plt.title(path)
        plt.legend()
        plt.savefig(path, transparent=True)
        if show:
            plt.show()
        plt.clf()

    def train(self, x_train, y_train, x_test, y_test, x_valid, y_valid, y_scale, batch_size=128, iters=10000, verbose=True):
        ''' Siyan added START '''
        # np.random.seed(1234)
        test_eval_count = 0
        valid_eval_count = 0
        test_best_iter_str = '0'
        valid_best_iter_str = '0'

        x_valid_input = tf.convert_to_tensor(x_valid, tf.float32)
        y_valid_input = tf.convert_to_tensor(y_valid, tf.float32)

        tic = time.time()
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, y_hat = self.run_train_step(x_input_batch, y_input_batch)

            if self.iter % 10 == 0:
                self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat)

            if self.iter % 100 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                mu, var, vloss, rmse, nll = self.evaluate(x_test_batch, y_test_batch)
                nll += np.log(y_scale[0,0])
                rmse *= y_scale[0,0]

                self.save_val_summary(vloss, x_test_batch, y_test_batch, mu, var)

                if rmse.numpy() < self.min_rmse:
                    self.min_rmse = rmse.numpy()
                    self.save(f"model_rmse_{self.iter}")

                if nll.numpy() < self.min_nll:
                    self.min_nll = nll.numpy()
                    self.save(f"model_nll_{self.iter}")

                if vloss.numpy() < self.min_vloss:
                    self.min_vloss = vloss.numpy()
                    self.save(f"model_vloss_{self.iter}")

               # if verbose: print("[{}] \t RMSE: {:.4f} \t NLL: {:.4f} \t train_loss: {:.4f} \t t: {:.2f} sec".format(self.iter, self.min_rmse, self.min_nll, loss, time.time()-tic))
               #  tic = time.time()


            ''' Siyan Test START --- (for entire test data instead of batch of test data)'''
            if self.iter % 10 == 0:
                x_test_input = tf.convert_to_tensor(x_test, tf.float32)
                y_test_input = tf.convert_to_tensor(y_test, tf.float32)
                test_mu, test_var, test_vloss, test_rmse, test_nll = self.evaluate(x_test_input, y_test_input)
                if test_eval_count == 0:
                    tmp_test_loss = test_vloss
                else:
                    if test_vloss < tmp_test_loss:
                        tmp_test_loss = test_vloss
                        print("[{}] Test loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} ".
                              format(self.iter, test_vloss, test_rmse.numpy(), test_nll.numpy()))
                        ### update the DataFrame
                        test_results_df = pd.DataFrame({
                            'test_x': list(x_test.flatten()),
                            'test_y': list(y_test.flatten()),
                            'test_mu': list(test_mu.numpy().flatten()),
                            'test_var': list(test_var.numpy().flatten())
                        })
                        test_best_iter_str = str(self.iter)
                test_eval_count += 1
                ''' Siyan Test END --- (for entire test data instead of batch of test data)'''

                ''' Siyan Test START --- (for entire validation data instead of batch of validation data)'''
                valid_mu, valid_var, valid_vloss, valid_rmse, valid_nll = self.evaluate(x_valid_input, y_valid_input)
                if valid_eval_count == 0:
                    tmp_valid_loss = valid_vloss
                else:
                    if valid_vloss < tmp_valid_loss:
                        tmp_valid_loss = valid_vloss
                        print("[{}] Validation loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} ".
                              format(self.iter, valid_vloss, valid_rmse.numpy(), valid_nll.numpy()))
                        ### update the DataFrame
                        valid_results_df = pd.DataFrame({
                            'valid_x': list(x_valid.flatten()),
                            'valid_y': list(y_valid.flatten()),
                            'valid_mu': list(valid_mu.numpy().flatten()),
                            'valid_var': list(valid_var.numpy().flatten())
                        })
                        valid_best_iter_str = str(self.iter)

                valid_eval_count += 1
                ''' Siyan Test END --- (for entire validation data instead of batch of validation data)'''

        test_results_df.to_csv(self.custom_best_results_dat_folder + "/best_test_results_iter_"+test_best_iter_str+".dat", sep=" ")
        print('--- Saved testing results to '+self.custom_best_results_dat_folder+'/best_test_results_iter_'+test_best_iter_str+".dat")
        valid_results_df.to_csv(self.custom_best_results_dat_folder + "/best_valid_results_iter_"+valid_best_iter_str+".dat", sep=" ")
        print('--- Saved validation results to '+self.custom_best_results_dat_folder +'/best_valid_results_iter_'+valid_best_iter_str+".dat")

        ''' Test/validation results calculation and plotting:
        (1) Best train results; (2) Best validation results. And compare to the original results
        '''
        load_test_df = pd.read_csv(self.custom_best_results_dat_folder + "/best_test_results_iter_"+test_best_iter_str+".dat", sep=" ")
        load_valid_df = pd.read_csv(self.custom_best_results_dat_folder + "/best_valid_results_iter_"+valid_best_iter_str+".dat", sep=" ")

        epistemic_test = load_test_df['test_mu'].values + load_test_df['test_var'].values
        epistemic_valid = load_valid_df['valid_mu'].values + load_valid_df['valid_var'].values

        valid_bounds = [[-7, +7]]
        self.plot_scatter_with_var_from_pandas(x_train, y_train, x_valid, y_valid,
                                               load_test_df['test_mu'].values, epistemic_test, path=self.custom_plot_folder+"/test_plot.pdf", n_stds=3, test_bounds=valid_bounds, show=True)
        self.plot_scatter_with_var_from_pandas(x_train, y_train, x_valid, y_valid,
                                               load_valid_df['valid_mu'].values, epistemic_valid, path=self.custom_plot_folder+"/valid_plot.pdf", n_stds=3, test_bounds=valid_bounds, show=True)


        return self.models, self.min_rmse, self.min_nll


# def plot_ensemble(models, save="ensemble", ext=".pdf"):
#     x_test_input = tf.convert_to_tensor(x_test, tf.float32)
#     preds = tf.stack([model(x_test_input, training=False) for model in models], axis=0) #forward pass
#     mus, sigmas = tf.split(preds, 2, axis=-1)
#
#     mean_mu = tf.reduce_mean(mus, axis=0)
#     epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(sigmas, axis=0)
#     plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)