from src.models.optimized_bnn import BayesianNN, DBNN
from src.models.deep_ensemble import DeepEnsemble
from src.attacks.point_attacks_jax import attack

from src.utils2 import plot_ppds

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted", font="serif")

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titleweight': 'bold',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'legend.fontsize': 12,
    'legend.frameon': False,
    'figure.dpi': 300,  
})

import numpyro
numpyro.set_host_device_count(8)

# load MNIST data without tensorflow
from sklearn.datasets import fetch_openml

# Load notMNIST data without labels
import os
from PIL import Image



def main():
    # Load MNIST from OpenML
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.values, mnist.target.values
    X = X / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype(int)

    # Split into training and testing datasets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Ruta de la carpeta con imágenes
    folder_path = '../data/notMNIST_small'

    # Lista para almacenar las imágenes
    images = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.png'):
                img_path = os.path.join(dirpath, filename)
                img = Image.open(img_path)
                img_gray = img.convert('L')
                img_array = np.array(img_gray)
                images.append(img_array)

    X_notmnist = np.array(images).reshape(-1, 28*28) / 255.0


    # Attack MNIST data to rise the entropy of the predictive distribution
    path = '/Users/pgarc/projects/AdvReg/src/models/weights/DE/mnist'
    model = DeepEnsemble(input_dim=X_train.shape[1], hidden_units=200, output_dim=10, num_models=20, model_type='mnist_mlp')
    model.load(path)

    def entropy(x, logits):
        pred = jax.nn.softmax(logits, axis=1).mean(axis=0)
        entr = (jax.scipy.special.entr(pred) / jax.numpy.log(2)).sum()
        return entr

    entropies = []
    for x in tqdm(X_test[:100]):
        x = jax.numpy.array(x.reshape(1,-1))
        G = 5 # we want to rise the entropy of the predictive distribution
        x_adv_values, loss_values, func_values = attack(x, model, G, func=entropy, samples_per_iteration=20)
        logits = model.sample_predictive_distribution(x_adv_values[-1], 20)
        pred = jax.nn.softmax(logits, axis=1).mean(axis=0)
        entropies.append((jax.scipy.special.entr(pred) / jax.numpy.log(2)).sum())

    sns.histplot(jnp.array(entropies), kde=True, bins=10, color='black', alpha=0.)
    plt.show()


if __name__ == '__main__':
    main()