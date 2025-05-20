# Evasion Attacks on Bayesian Models

This repository contains the code needed to reproduce all the experiments in the **Evasion Attacks Against Bayesian Predictive Models** paper.

> There is an increasing interest in analyzing the behavior of machine learning systems against adversarial attacks. However, most of the research in adversarial machine learning has focused on studying weaknesses against evasion or poisoning attacks to predictive models in classical setups, with the susceptibility of Bayesian predictive models to attacks remaining underexplored. This paper introduces a general methodology for designing optimal evasion attacks against such models. We investigate two adversarial objectives: perturbing specific point predictions and altering the entire posterior predictive distribution. For both scenarios, we propose novel gradient-based attacks and study their implementation and properties in various computational setups.

The code is written Python. A conda environment contains all necessary dependencies. It can be installed using

`conda env create -f advReg.yml`

And activated through 

`conda activate advReg`

In addition, the `AdvReg` package must be installed running the following in the root directory:

`pip install -e .`

### Notebooks to reproduce experiments:

- Section 5.1: Notebooks toy_distr.ipynb and toy_point.ipynb.

- Section 5.2: Notebooks three_datasets_distr.ipynb and three_datasets_point.ipynb.

- Section 5.3: Notebooks test_distr_bnn_jax.ipynb and test_point_bnn.ipynb.

- Section 5.4: Notebooks test__mnist_jax_VI.ipynb.

- Section 5.5: Notebooks test_distr_bnn_jax_gray.ipynb and test_point_bnn_gray.ipynb.

- Supplementary Section E: Notebooks toy_distr_dependent.ipynb, toy_distr.ipynb and toy_point.ipynb.

- Supplementary Section F: Notebooks three_datasets_distr.ipynb and three_datasets_point.ipynb.

- Supplementary Section G: Notebooks test_distr_bnn_jax.ipynb and test_point_bnn.ipynb.

- Supplementary Section H: Notebooks test__mnist_jax_VI.ipynb and test_point_de_jax.ipynb.

- Supplementary Section I: Notebooks test_distr_bnn_jax_gray.ipynb and test_point_bnn_gray.ipynb.
