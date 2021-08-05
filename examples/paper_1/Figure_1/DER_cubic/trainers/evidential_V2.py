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

class Evidential:
    def __init__(self, model, opts, dataset="", learning_rate=1e-3, lam=0.0, epsilon=1e-2, maxi_rate=1e-4, tag="", custom_plot_folder="", custom_best_results_dat_folder=""):
        self.nll_loss_function = edl.losses.NIG_NLL
        self.reg_loss_function = edl.losses.NIG_Reg

        self.model = model
        self.learning_rate = learning_rate
        self.maxi_rate = maxi_rate

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.lam = tf.Variable(lam)

        self.epsilon = epsilon

        self.min_rmse = self.running_rmse = float('inf')
        self.min_nll = self.running_nll = float('inf')
        self.min_vloss = self.running_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save','{}_{}_{}_{}'.format(current_time, dataset, trainer, tag))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.custom_plot_folder = custom_plot_folder
        Path(self.custom_plot_folder).mkdir(parents=True, exist_ok=True)
        ''' Create 4 folders for valid/test plots for Epistemic/Aleatoric uncertainty'''
        Path(self.custom_plot_folder + '/valid_epistemic').mkdir(parents=True, exist_ok=True)
        Path(self.custom_plot_folder + '/valid_aleatoric').mkdir(parents=True, exist_ok=True)
        Path(self.custom_plot_folder + '/test_epistemic').mkdir(parents=True, exist_ok=True)
        Path(self.custom_plot_folder + '/test_aleatoric').mkdir(parents=True, exist_ok=True)

        self.custom_best_results_dat_folder = custom_best_results_dat_folder
        Path(self.custom_best_results_dat_folder).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join('logs', '{}_{}_{}_{}_train'.format(current_time, dataset, trainer, tag))
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs', '{}_{}_{}_{}_val'.format(current_time, dataset, trainer, tag))
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def loss_function(self, y, mu, v, alpha, beta, reduce=True, return_comps=False):
        nll_loss = self.nll_loss_function(y, mu, v, alpha, beta, reduce=reduce)
        reg_loss = self.reg_loss_function(y, mu, v, alpha, beta, reduce=reduce)
        loss = nll_loss + self.lam * (reg_loss - self.epsilon)
        # loss = nll_loss

        return (loss, (nll_loss, reg_loss)) if return_comps else loss

    @tf.function
    def run_train_step(self, x, y):
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)
            loss, (nll_loss, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

        grads = tape.gradient(loss, self.model.trainable_variables) #compute gradient
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.lam = self.lam.assign_add(self.maxi_rate * (reg_loss - self.epsilon)) #update lambda

        return loss, nll_loss, reg_loss, mu, v, alpha, beta

    @tf.function
    def evaluate(self, x, y):
        outputs = self.model(x, training=False)
        mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)

        rmse = edl.losses.RMSE(y, mu)
        loss, (nll, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

        return mu, v, alpha, beta, loss, rmse, nll, reg_loss

    def normalize(self, x):
        return tf.divide(tf.subtract(x, tf.reduce_min(x)),
               tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))


    def save_train_summary(self, loss, x, y, y_hat, v, alpha, beta):
        # np.random.seed(1234)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, y_hat)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(self.loss_function(y, y_hat, v, alpha, beta)), step=self.iter)
            # pass
            idx = np.random.choice(int(tf.shape(x)[0]), 9)  ###  !!! give different model predictions !!!!!
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(y_hat,idx).numpy())], max_outputs=1, step=self.iter)

    # ''' Add additional saving for predicted y_hat and y_var '''
    # def save_train_summary_scalar(self, loss, x, y, y_hat, v, alpha, beta):
    #     with self.train_summary_writer.as_default():
    #         tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, y_hat)), step=self.iter)
    #         tf.summary.scalar('loss', tf.reduce_mean(self.loss_function(y, y_hat, v, alpha, beta)), step=self.iter)
    #         # tf.summary.scalar('y_hat_2', y_hat, step=self.iter)
    #         # var = beta / (v * (alpha - 1))
    #         # tf.summary.scalar('var', var, step=self.iter)
    #         idx = np.random.choice(int(tf.shape(x)[0]), 9)
    #         if tf.shape(x).shape == 4:
    #             tf.summary.image("x", [gallery(tf.gather(x, idx).numpy())], max_outputs=1, step=self.iter)
    #
    #         if tf.shape(y).shape == 4:
    #             tf.summary.image("y", [gallery(tf.gather(y, idx).numpy())], max_outputs=1, step=self.iter)
    #             tf.summary.image("y_hat", [gallery(tf.gather(y_hat, idx).numpy())], max_outputs=1, step=self.iter)

    def save_val_summary(self, loss, x, y, mu, v, alpha, beta):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, mu)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(self.loss_function(y, mu, v, alpha, beta)), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)

            # var = beta / (v * (alpha - 1))
            # print('--Type of v: {}'.format(type(v)))
            # print('--Type of alpha: {}'.format(type(alpha)))
            # print('--Type of beta: {}'.format(type(beta)))
            # print('--Type of var: {}'.format(type(var)))

            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(mu,idx).numpy())], max_outputs=1, step=self.iter)
                var = beta/(v*(alpha-1))
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
        self.model.save(os.path.join(self.save_dir, "{}.h5".format(name)))

    def update_running(self, previous, current, alpha=0.0):
        if previous == float('inf'):
            new = current
        else:
            new = alpha*previous + (1-alpha)*current
        return new


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
        # print('asdfsdafdsafdsafsdafdsaf')
        # print(path)
        plt.savefig(path, transparent=True)
        if show:
            plt.show()
        plt.clf()

    ''' Siyan added for testing plot V2 '''
    def plot_scatter_with_var_from_pandas_v2(self, iter, x_train, y_train, x_test, y_test, mu, var, path, n_stds=3, test_bounds=[[-7, +7]], evalType='valid', uq='epistemic', show=True, save=False):
        # print(path)
        plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label='Train (x_train vs y_train)')
        idxx = np.argsort(x_test.flatten())
        x_test = x_test.flatten()[idxx]
        y_test = y_test.flatten()[idxx]
        # mu = mu.numpy().flatten()[idxx]
        mu = mu.flatten()[idxx]
        var = var.flatten()[idxx]
        for k in np.linspace(0, n_stds, 4):
            if k == 0:
                plt.fill_between(x_test, (mu - k * var), (mu + k * var), alpha=0.3, edgecolor=None,
                                 facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1, label='Unc.')
            else:
                plt.fill_between(x_test, (mu - k * var), (mu + k * var), alpha=0.3, edgecolor=None,
                                 facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)

        # idxx = np.argsort(x_test.flatten())
        # print(idxx)
        # plt.plot(x_test.flatten()[idxx], y_test.flatten()[idxx], 'r--', zorder=2, label='True (x_test vs y_test)')
        # plt.plot(x_test.flatten()[idxx], mu.flatten()[idxx], color='#007cab', zorder=3, label='Pred (x_test vs mu)')

        if evalType == 'valid':
            plt.plot(x_test, y_test, 'r--', zorder=2, label='True (x_valid vs y_valid)')
            plt.plot(x_test, mu, color='#007cab', zorder=3, label='Pred (x_valid vs mu)')
        elif evalType == 'test':
            plt.plot(x_test, y_test, 'r--', zorder=2, label='True (x_test vs y_test)')
            plt.plot(x_test, mu, color='#007cab', zorder=3, label='Pred (x_test vs mu)')

        plt.gca().set_xlim(*test_bounds)
        plt.gca().set_ylim(-150, 150)
        plt.title(path)
        if uq == 'epistemic':
            plt.suptitle('Epistemic uncertainty of {} data (Iters: {})\n'.format(evalType, iter), fontsize=18)
        elif uq == 'aleatoric':
            plt.suptitle('Aleatoric uncertainty of {} data (Iters: {})\n'.format(evalType, iter), fontsize=18)

        plt.legend()
        if save:
            plt.savefig(path)
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

        ''' Siyan added END '''
        tic = time.time()
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, nll_loss, reg_loss, y_hat, v, alpha, beta = self.run_train_step(x_input_batch, y_input_batch)

            # if self.iter % 10 == 0:
            #     # np.random.seed(1234)
            #     self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)
            #     # self.save_train_summary_scalar(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)
            #
            if self.iter % 10 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                mu, v, alpha, beta, vloss, rmse, nll, reg_loss = self.evaluate(x_test_batch, y_test_batch)

                nll += np.log(y_scale[0, 0])
                rmse *= y_scale[0, 0]

                ## validation summary
                self.save_val_summary(vloss, x_test_batch, y_test_batch, mu, v, alpha, beta)

                self.running_rmse = self.update_running(self.running_rmse, rmse.numpy())
                if self.running_rmse < self.min_rmse:
                    self.min_rmse = self.running_rmse
                    # self.save(f"model_rmse_{self.iter}")

                self.running_nll = self.update_running(self.running_nll, nll.numpy())
                if self.running_nll < self.min_nll:
                    self.min_nll = self.running_nll
                    # self.save(f"model_nll_{self.iter}")

                self.running_vloss = self.update_running(self.running_vloss, vloss.numpy())
                if self.running_vloss < self.min_vloss:
                    self.min_vloss = self.running_vloss
                    # self.save(f"model_vloss_{self.iter}")

                # if verbose: print(
                #     "[A {}]  RMSE: {:.4f} \t NLL: {:.4f} \t loss: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f} \t t: {:.2f} sec".format(
                #         self.iter, self.min_rmse, self.min_nll, vloss, reg_loss.numpy().mean(), self.lam.numpy(),
                #         time.time() - tic))



                # ### Loss function for reference
                # def loss_function(self, y, mu, v, alpha, beta, reduce=True, return_comps=False):
                #     nll_loss = self.nll_loss_function(y, mu, v, alpha, beta, reduce=reduce)
                #     reg_loss = self.reg_loss_function(y, mu, v, alpha, beta, reduce=reduce)
                #     loss = nll_loss + self.lam * (reg_loss - self.epsilon)
                #     return (loss, (nll_loss, reg_loss)) if return_comps else loss


            if self.iter % 10 == 0:
                # self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)


                ''' Siyan Test START --- (for entire test data instead of batch of test data)'''
                x_test_input = tf.convert_to_tensor(x_test, tf.float32)
                y_test_input = tf.convert_to_tensor(y_test, tf.float32)
                tmp_outputs_1 = self.model(x_test_input, training=False)
                test_mu, test_v, test_alpha, test_beta = tf.split(tmp_outputs_1, 4, axis=1)
                test_rmse = edl.losses.RMSE(y_test_input, test_mu)

                ### Calculate loss for all training data instead of a batch of training data
                test_loss, (test_nll_loss, test_reg_loss) = self.loss_function(y_test_input, test_mu, test_v, test_alpha, test_beta, return_comps=True)
                # print(test_reg_loss)
                if test_eval_count == 0:
                    tmp_test_loss = test_loss
                else:
                    if test_loss < tmp_test_loss:
                        tmp_test_loss = test_loss
                        print("[{}] Test loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                              format(self.iter, test_loss, test_rmse.numpy(), test_nll_loss.numpy(), test_reg_loss.numpy(), self.lam.numpy()))
                        ### update the DataFrame
                        test_results_df = pd.DataFrame({
                            'test_x': list(x_test.flatten()),
                            'test_y': list(y_test.flatten()),
                            'test_mu': list(test_mu.numpy().flatten()),
                            'test_v': list(test_v.numpy().flatten()),
                            'test_alpha': list(test_alpha.numpy().flatten()),
                            'test_beta': list(test_beta.numpy().flatten()),
                        })
                        test_best_iter_str = str(self.iter)
                test_eval_count += 1
                ''' Siyan Test END --- (for entire test data instead of batch of test data)'''

                ''' Siyan Test START --- (for entire validation data instead of batch of validation data)'''
                tmp_outputs_2 = self.model(x_valid_input, training=False)
                valid_mu, valid_v, valid_alpha, valid_beta = tf.split(tmp_outputs_2, 4, axis=1)
                valid_rmse = edl.losses.RMSE(y_valid_input, valid_mu)

                valid_loss, (valid_nll_loss, valid_reg_loss) = self.loss_function(y_valid_input, valid_mu, valid_v,
                                                                                  valid_alpha, valid_beta,
                                                                                  return_comps=True)

                # print("[{}] Valid loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                #       format(self.iter, valid_loss, valid_rmse.numpy(), valid_nll_loss.numpy(), valid_reg_loss.numpy(),
                #              self.lam.numpy()))

                if valid_eval_count == 0:
                    tmp_valid_loss = valid_loss
                    # tmp_valid_rmse = valid_rmse
                else:
                    if valid_loss < tmp_valid_loss:
                    # if valid_rmse < tmp_valid_rmse:
                        tmp_valid_loss = valid_loss
                        # tmp_valid_rmse = valid_rmse
                        print("[{}] Valid loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                              format(self.iter, valid_loss, valid_rmse.numpy(), valid_nll_loss.numpy(), valid_reg_loss.numpy(), self.lam.numpy()))
                        ### update the DataFrame
                        valid_results_df = pd.DataFrame({
                            'valid_x': list(x_valid.flatten()),
                            'valid_y': list(y_valid.flatten()),
                            'valid_mu': list(valid_mu.numpy().flatten()),
                            'valid_v': list(valid_v.numpy().flatten()),
                            'valid_alpha': list(valid_alpha.numpy().flatten()),
                            'valid_beta': list(valid_beta.numpy().flatten()),
                        })
                        valid_best_iter_str = str(self.iter)

                # valid_results_df = pd.DataFrame({
                #     'valid_x': list(x_valid.flatten()),
                #     'valid_y': list(y_valid.flatten()),
                #     'valid_mu': list(valid_mu.numpy().flatten()),
                #     'valid_v': list(valid_v.numpy().flatten()),
                #     'valid_alpha': list(valid_alpha.numpy().flatten()),
                #     'valid_beta': list(va.lid_beta.numpy().flatten()),
                # })
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

        epistemic_test = np.sqrt(load_test_df['test_beta'].values / (load_test_df['test_v'].values * (load_test_df['test_alpha'].values - 1)))
        epistemic_test = np.minimum(epistemic_test, 1e3)  # clip the unc for vis

        print(load_test_df)
        epistemic_valid = np.sqrt(load_valid_df['valid_beta'].values / (load_valid_df['valid_v'].values * (load_valid_df['valid_alpha'].values - 1)))
        epistemic_valid = np.minimum(epistemic_valid, 1e3)  # clip the unc for vis

        valid_bounds = [[-7, +7]]
        self.plot_scatter_with_var_from_pandas(x_train, y_train, x_valid, y_valid,
                                               load_test_df['test_mu'].values, epistemic_test, path=self.custom_plot_folder+"/test_plot.pdf", n_stds=3, test_bounds=valid_bounds, show=True)
        self.plot_scatter_with_var_from_pandas(x_train, y_train, x_valid, y_valid,
                                               load_valid_df['valid_mu'].values, epistemic_valid, path=self.custom_plot_folder+"/valid_plot.pdf", n_stds=3, test_bounds=valid_bounds, show=True)


        return self.model, self.min_rmse, self.min_nll


    # def train(self, x_train, y_train, x_test, y_test, x_valid, y_valid, y_scale, batch_size=128, iters=10000, verbose=True):
    def train_v2(self, x_train, y_train, x_valid, y_valid, x_test, y_test, y_scale, batch_size=128, iters=10000, verbose=True):
        ''' Siyan added START '''
        # np.random.seed(1234)
        valid_eval_count = 0
        valid_best_iter_str = '0'
        test_eval_count = 0
        test_best_iter_str = '0'

        x_test_input = tf.convert_to_tensor(x_test, tf.float32)
        y_test_input = tf.convert_to_tensor(y_test, tf.float32)

        ''' Siyan added END '''
        tic = time.time()
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, nll_loss, reg_loss, y_hat, v, alpha, beta = self.run_train_step(x_input_batch, y_input_batch)

            if self.iter % 10 == 0:
                # np.random.seed(1234)
                self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)
                # self.save_train_summary_scalar(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)

            if self.iter % 10 == 0:
                x_valid_batch, y_valid_batch = self.get_batch(x_valid, y_valid, min(100, x_valid.shape[0]))
                mu, v, alpha, beta, vloss, rmse, nll, reg_loss = self.evaluate(x_valid_batch, y_valid_batch)

                nll += np.log(y_scale[0, 0])
                rmse *= y_scale[0, 0]

                ## validation summary
                self.save_val_summary(vloss, x_valid_batch, y_valid_batch, mu, v, alpha, beta)

                self.running_rmse = self.update_running(self.running_rmse, rmse.numpy())
                if self.running_rmse < self.min_rmse:
                    self.min_rmse = self.running_rmse
                    # self.save(f"model_rmse_{self.iter}")

                self.running_nll = self.update_running(self.running_nll, nll.numpy())
                if self.running_nll < self.min_nll:
                    self.min_nll = self.running_nll
                    # self.save(f"model_nll_{self.iter}")

                self.running_vloss = self.update_running(self.running_vloss, vloss.numpy())
                if self.running_vloss < self.min_vloss:
                    self.min_vloss = self.running_vloss
                    # self.save(f"model_vloss_{self.iter}")

                # if verbose: print(
                #     "[A {}]  RMSE: {:.4f} \t NLL: {:.4f} \t loss: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f} \t t: {:.2f} sec".format(
                #         self.iter, self.min_rmse, self.min_nll, vloss, reg_loss.numpy().mean(), self.lam.numpy(),
                #         time.time() - tic))

                # ### Loss function for reference
                # def loss_function(self, y, mu, v, alpha, beta, reduce=True, return_comps=False):
                #     nll_loss = self.nll_loss_function(y, mu, v, alpha, beta, reduce=reduce)
                #     reg_loss = self.reg_loss_function(y, mu, v, alpha, beta, reduce=reduce)
                #     loss = nll_loss + self.lam * (reg_loss - self.epsilon)
                #     return (loss, (nll_loss, reg_loss)) if return_comps else loss


            if self.iter % 10 == 0 or self.iter < 50 or self.iter >= iters - 50:
                # self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat, v, alpha, beta)

                ''' Siyan Test START --- (for all validation dat)'''
                x_valid_input = tf.convert_to_tensor(x_valid, tf.float32)
                y_valid_input = tf.convert_to_tensor(y_valid, tf.float32)
                tmp_outputs_1 = self.model(x_valid_input, training=False)
                valid_mu, valid_v, valid_alpha, valid_beta = tf.split(tmp_outputs_1, 4, axis=1)
                valid_rmse = edl.losses.RMSE(y_valid_input, valid_mu)

                ### Calculate loss for batch validation data
                valid_loss, (valid_nll_loss, valid_reg_loss) = self.loss_function(y_valid_input, valid_mu, valid_v, valid_alpha, valid_beta, return_comps=True)


                ### Calculate Epistemic, Aleatoric uncertainty for validation data directly and plot
                ''' epistemic (Var[mu]) = beta/ (v(alpha - 1)) '''
                epistemic_valid = np.sqrt(valid_beta / (valid_v * (valid_alpha - 1)))
                aleatoric_sigma_valid = np.sqrt((epistemic_valid ** 2) * valid_v)
                epistemic_valid = np.minimum(epistemic_valid, 1e3)  # clip the unc for vis
                aleatoric_sigma_valid = np.minimum(aleatoric_sigma_valid, 1e3)  # clip the unc for vis
                test_bounds = [[-7, +7]]

                ### x_train, y_train: np_arr (size, 1)
                ### x_valid, y_valid: np_arr (size, 1)
                ### valid_mu: tf tensor (size, 1)
                ### epistemic_valid: np_arr (size, 1)
                valid_mu_np = valid_mu.numpy()
                self.plot_scatter_with_var_from_pandas_v2(self.iter, x_train, y_train, x_valid, y_valid, valid_mu_np, epistemic_valid,
                                                          path=self.custom_plot_folder+"/valid_epistemic"+"/valid_plot_iter_{}".format(self.iter)+".png",
                                                          n_stds=3, test_bounds=test_bounds, evalType='valid', uq='epistemic', show=False, save=True)
                self.plot_scatter_with_var_from_pandas_v2(self.iter, x_train, y_train, x_valid, y_valid, valid_mu_np, aleatoric_sigma_valid,
                                                          path=self.custom_plot_folder+"/valid_aleatoric"+"/valid_plot_iter_{}".format(self.iter)+".png",
                                                          n_stds=3, test_bounds=test_bounds, evalType='valid', uq='aleatoric', show=False, save=True)

                # # print(valid_reg_loss)
                if valid_eval_count == 0:
                    tmp_valid_loss = valid_loss
                else:
                    if valid_loss < tmp_valid_loss:
                        tmp_valid_loss = valid_loss
                        print("[{}] Valid loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                              format(self.iter, valid_loss, valid_rmse.numpy(), valid_nll_loss.numpy(), valid_reg_loss.numpy(), self.lam.numpy()))
                        ### update the DataFrame
                        valid_results_df = pd.DataFrame({
                            'valid_x': list(x_valid.flatten()),
                            'valid_y': list(y_valid.flatten()),
                            'valid_mu': list(valid_mu.numpy().flatten()),
                            'valid_v': list(valid_v.numpy().flatten()),
                            'valid_alpha': list(valid_alpha.numpy().flatten()),
                            'valid_beta': list(valid_beta.numpy().flatten()),
                        })
                        valid_best_iter_str = str(self.iter)
                valid_eval_count += 1
                ''' Siyan Test END --- (for all validation data)'''

                ''' Siyan Test START --- (for entire testing data)'''
                tmp_outputs_2 = self.model(x_test_input, training=False)
                test_mu, test_v, test_alpha, test_beta = tf.split(tmp_outputs_2, 4, axis=1)
                test_rmse = edl.losses.RMSE(y_test_input, test_mu)

                test_loss, (test_nll_loss, test_reg_loss) = self.loss_function(y_test_input, test_mu, test_v,
                                                                                  test_alpha, test_beta,
                                                                                  return_comps=True)

                ### Calculate Epistemic, Aleatoric uncertainty for test data directly and plot
                ''' epistemic (Var[mu]) = beta/ (v(alpha - 1)) '''
                epistemic_test = np.sqrt(test_beta / (test_v * (test_alpha - 1)))
                aleatoric_sigma_test = np.sqrt((epistemic_test ** 2) * test_v)
                epistemic_test = np.minimum(epistemic_test, 1e3)  # clip the unc for vis
                aleatoric_sigma_test = np.minimum(aleatoric_sigma_test, 1e3)  # clip the unc for vis
                test_bounds = [[-7, +7]]

                test_mu_np = test_mu.numpy()
                self.plot_scatter_with_var_from_pandas_v2(self.iter, x_train, y_train, x_test, y_test, test_mu_np, epistemic_test,
                                                          path=self.custom_plot_folder+"/test_epistemic"+"/test_plot_iter_{}".format(self.iter)+".png",
                                                          n_stds=3, test_bounds=test_bounds, evalType='test', uq='epistemic', show=False, save=True)
                self.plot_scatter_with_var_from_pandas_v2(self.iter, x_train, y_train, x_test, y_test, test_mu_np, aleatoric_sigma_test,
                                                          path=self.custom_plot_folder+"/test_aleatoric"+"/test_plot_iter_{}".format(self.iter)+".png",
                                                          n_stds=3, test_bounds=test_bounds, evalType='test', uq='aleatoric', show=False, save=True)

                # print("[{}] Valid loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                #       format(self.iter, valid_loss, valid_rmse.numpy(), valid_nll_loss.numpy(), valid_reg_loss.numpy(),
                #              self.lam.numpy()))

                # test_mu_np = test_mu.numpy()
                # self.plot_scatter_with_var_from_pandas_v2(self.iter, x_train, y_train, x_test, y_test, test_mu_np, epistemic_test,
                #                                           path=self.custom_plot_folder+"/test_plot_iter_{}".format(self.iter)+".png",
                #                                           n_stds=3, test_bounds=test_bounds, evalType='test', uq='epistemic', show=False, save=True)

                if test_eval_count == 0:
                    tmp_test_loss = test_loss
                    # tmp_valid_rmse = valid_rmse
                else:
                    if test_loss < tmp_test_loss:
                    # if valid_rmse < tmp_valid_rmse:
                        tmp_test_loss = test_loss
                        # tmp_valid_rmse = valid_rmse
                        print("[{}] Test loss: {:.6f} \t RMSE: {:.4f} \t NLL: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".
                              format(self.iter, test_loss, test_rmse.numpy(), test_nll_loss.numpy(), test_reg_loss.numpy(), self.lam.numpy()))
                        ### update the DataFrame
                        test_results_df = pd.DataFrame({
                            'test_x': list(x_test.flatten()),
                            'test_y': list(y_test.flatten()),
                            'test_mu': list(test_mu.numpy().flatten()),
                            'test_v': list(test_v.numpy().flatten()),
                            'test_alpha': list(test_alpha.numpy().flatten()),
                            'test_beta': list(test_beta.numpy().flatten()),
                        })
                        test_best_iter_str = str(self.iter)
                test_eval_count += 1

                ''' Siyan Test END --- (for entire test data)'''

        valid_results_df.to_csv(self.custom_best_results_dat_folder + "/best_valid_results_iter_"+valid_best_iter_str+".dat", sep=" ")
        print('--- Saved validation results to '+self.custom_best_results_dat_folder +'/best_valid_results_iter_'+valid_best_iter_str+".dat")
        test_results_df.to_csv(self.custom_best_results_dat_folder + "/best_test_results_iter_"+test_best_iter_str+".dat", sep=" ")
        print('--- Saved testing results to '+self.custom_best_results_dat_folder+'/best_test_results_iter_'+test_best_iter_str+".dat")


        ''' Validation/test results calculation and plotting:
        (1) Best validation results; (2) Best testing results. And compare to the original results
        '''
        load_valid_df = pd.read_csv(self.custom_best_results_dat_folder + "/best_valid_results_iter_"+valid_best_iter_str+".dat", sep=" ")
        load_test_df = pd.read_csv(self.custom_best_results_dat_folder + "/best_test_results_iter_"+test_best_iter_str+".dat", sep=" ")

        ''' epistemic (Var[mu]) = beta/ (v(alpha - 1)) '''
        epistemic_valid = np.sqrt(load_valid_df['valid_beta'].values / (load_valid_df['valid_v'].values * (load_valid_df['valid_alpha'].values - 1)))
        epistemic_valid = np.minimum(epistemic_valid, 1e3)  # clip the unc for vis

        epistemic_test = np.sqrt(load_test_df['test_beta'].values / (load_test_df['test_v'].values * (load_test_df['test_alpha'].values - 1)))
        epistemic_test = np.minimum(epistemic_test, 1e3)  # clip the unc for vis

        ''' aleatoric (E[sigma^2]) = beta / (alpha - 1)'''
        # aleatoric_valid_sigma = np.sqrt()

        # print(load_test_df)

        # valid_bounds = [[-4, +4]]
        test_bounds = [[-7, +7]]

        # print(x_train.shape)
        # print(y_train.shape)
        # print(x_valid.shape)
        # print(y_valid.shape)
        # print(load_valid_df['valid_mu'].values.shape)
        # print(epistemic_valid.shape)


        # self.plot_scatter_with_var_from_pandas_v2(x_train, y_train, x_valid, y_valid,
        #                                        load_valid_df['valid_mu'].values, epistemic_valid, path=self.custom_plot_folder+"/valid_plot.pdf", n_stds=3, test_bounds=test_bounds, show=True)

        #
        # self.plot_scatter_with_var_from_pandas_v2(x_train, y_train, x_test, y_test,
        #                                        load_test_df['test_mu'].values, epistemic_test, path=self.custom_plot_folder+"/test_plot.pdf", n_stds=3, test_bounds=test_bounds, show=True)

        return self.model, self.min_rmse, self.min_nll
