import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
import os

import trainers
import models


def generate_cubic_asymmetric(xdata, noise=False):
    N = len(xdata)
    data_noise = np.reshape(np.random.normal(0, 1, N), xdata.shape)
    data_noise[data_noise > 0] = data_noise[data_noise > 0] * 10.0
    data_noise[data_noise < 0] = data_noise[data_noise < 0] * 2.0
    if noise:
        ydata = xdata ** 3 + data_noise  ## y=x^3+data_noise
    else:
        ydata = xdata ** 3
    return ydata


seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

save_fig_dir = "./figs/toy"
batch_size = 256 #128
iterations = 20000
show = True

noise_changing = False
train_bounds = [[-4, 4]]
x_train = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in train_bounds]).reshape(-1,1)
y_train = generate_cubic_asymmetric(x_train, noise=True)

test_bounds = [[-7,+7]]
x_test = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in test_bounds]).reshape(-1,1)
y_test = generate_cubic_asymmetric(x_test, noise=False)

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

#### Different toy configurations to train and plot
def evidence_reg_1_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=1)
    trainer = trainer_obj(model, opts, learning_rate=1e-4, lam=1e-3, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_1_layers_100_neurons"))

### Main file to run the different methods and compare results ###
if __name__ == "__main__":
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)
    evidence_reg_1_layers_100_neurons()

    print(f"Done! Figures saved to {save_fig_dir}")
