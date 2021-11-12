import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
from scipy import stats
import os

import trainers
import models


# def generate_cubic_asymmetric(xdata, noise=False):
#     N = len(xdata)
#     data_noise = np.reshape(np.random.normal(0, 1, N), xdata.shape)
#     data_noise[data_noise > 0] = data_noise[data_noise > 0] * 10.0
#     data_noise[data_noise < 0] = data_noise[data_noise < 0] * 2.0
#     if noise:
#         ydata = xdata ** 3 + data_noise  ## y=x^3+data_noise
#     else:
#         ydata = xdata ** 3
#     return ydata

# def target_fun(x):
#     y = x**3
#     y = torch.sum(y, dim=1, keepdim=True)/10.0 + 5.0*torch.randn(x.size(0), device=device).unsqueeze(1)
#     return y


def target_fun(x):
    y = x ** 3
    y = tf.reduce_sum(y, axis=1, keepdims=True)/10.0 + 1.0*tf.random.normal([x.shape[0], 1])
    return y




seed = 1234567
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

save_fig_dir = "./figs/toy"
batch_size = 256 #128
iterations = 3000
show = True

Npar = 10
Ntrain = 5000
Nout = 1


# x_train = tf.random.uniform(shape=(Ntrain, Npar))*4.0-2.0
x_train = tf.random.normal(shape=(Ntrain, Npar))
y_train = target_fun(x_train)

x_train = x_train.numpy()
y_train = y_train.numpy()

Ntest = 5000
# x_test = tf.random.uniform(shape=(Ntest, Npar)) + 2.0
# x_test = tf.random.uniform(shape=(Ntest, Npar))
# x_test = np.random.uniform(size=(Ntest,Npar))
# x_test[:,1] = x_test[:,1] + 3.5
x_test = np.random.normal(size=(Ntest,Npar)) + 2.0
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = target_fun(x_test)

x_test = x_test.numpy()
y_test = y_test.numpy()


### Plotting helper functions ###
def plot_scatter_with_var(mu, var, path, n_stds=3):
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0)
    k=1.96
    plt.fill_between(x_test[:,0], (mu-k*var)[:,0], (mu+k*var)[:,0], alpha=0.3, edgecolor=None, facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)

    plt.plot(x_test, y_test, 'r--', zorder=2)
    plt.plot(x_test, mu, color='#007cab', zorder=3)
    plt.gca().set_xlim(*test_bounds)
    plt.gca().set_ylim(-150,150)
    plt.title(path)
    plt.savefig(path, transparent=True)

    np.savetxt('EvidentialDL_cubic.dat', np.vstack((x_test[:,0], mu[:,0], var[:,0])))


    if show:
        plt.show()
    plt.clf()

def plot_ng(model, save="ng", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    outputs = model(x_test_input)
    mu, v, alpha, beta = tf.split(outputs, 4, axis=1)
    epistemic = np.sqrt(beta/(v*(alpha-1)))
    epistemic = np.minimum(epistemic, 1e3) # clip the unc for vis



    plot_scatter_with_var(mu, epistemic, path=save+ext, n_stds=3)


def analysis(model):
    x_train_input = tf.convert_to_tensor(x_train, tf.float32)
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)

    train_outputs = model(x_train_input)
    test_outputs = model(x_test_input)

    mu_train, v_train, alpha_train, beta_train = tf.split(train_outputs, 4, axis=1)
    mu_test, v_test, alpha_test, beta_test = tf.split(test_outputs, 4, axis=1)

    epistemic_train = np.sqrt(beta_train/(v_train*(alpha_train-1)))
    epistemic_train = np.minimum(epistemic_train, 1e3) # clip the unc for vis

    epistemic_test = np.sqrt(beta_test/(v_test*(alpha_test-1)))
    epistemic_test = np.minimum(epistemic_test, 1e3) # clip the unc for vis

    print(mu_train.shape)
    print(mu_test.shape)

    print(epistemic_train.shape)
    print(epistemic_test.shape)

    train_PIW_arr = 2*1.64*epistemic_train
    test_PIW_arr = 2*1.64*epistemic_test

    train_PIW_arr = train_PIW_arr.flatten()
    test_PIW_arr = test_PIW_arr.flatten()


    np.savetxt('DER_MPIW_train_bias.dat', train_PIW_arr)
    np.savetxt('DER_MPIW_test_bias.dat', test_PIW_arr)

    kde_train = stats.gaussian_kde(train_PIW_arr)
    kde_test = stats.gaussian_kde(test_PIW_arr)

    x1 = np.linspace(train_PIW_arr.min(), train_PIW_arr.max(), 100)
    p1 = kde_train(x1)

    x2 = np.linspace(test_PIW_arr.min(), test_PIW_arr.max(), 100)
    p2 = kde_test(x2)

    plt.plot(x1,p1, label='train')
    plt.plot(x2,p2, label='test')
    plt.legend()
    plt.savefig('DER_cubic10D_bias.png')
    plt.show()

    ###------- Option I: calculate confidence interval
    conf_score = kde_train(test_PIW_arr)/p1.max()
    print('--Conf_score mean: {}, STD: {}'.format(np.mean(conf_score), np.std(conf_score)))



#### Different toy configurations to train and plot
def evidence_reg_1_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=10, num_neurons=30, num_layers=1)
    trainer = trainer_obj(model, opts, learning_rate=1e-4, lam=1e-3, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    analysis(model)
    # plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_1_layers_100_neurons"))

### Main file to run the different methods and compare results ###
if __name__ == "__main__":
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)
    evidence_reg_1_layers_100_neurons()

    print(f"Done! Figures saved to {save_fig_dir}")
