# Evasion Attacks on Bayesian Regression

This repository contains the code needed to reproduce all the experiments in the **Evasion Attacks Against Bayesian Regression Models** paper.

> Machine learning systems are increasingly exposed to adversarial attacks. While much of the research in adversarial machine learning has focused on studying the weaknesses against evasion attacks against classification models in classical setups, the susceptibility of Bayesian regression models to attacks remains underexplored. This paper introduces a general methodology for designing optimal evasion attacks against such models.  We investigate two adversarial objectives: perturbing specific point predictions and altering the entire posterior predictive distribution. For both scenarios, we propose gradient-based attacks that are applicable even when the posterior predictive distribution lacks a closed-form solution and is accessible only through Markov Chain Monte Carlo sampling.

The code is written Python. A conda environment contains all necessary dependencies. It can be installed using

`conda env create -f advReg.yml`

And activated through 

`conda activate advReg`

In addition, the `AdvReg` package must be installed running the following in the root directory:

`pip install -e .`

### Notebooks to reproduce experiments:

- Section 5.1: Notebooks toy_distr.ipynb and toy_point.ipynb.

- Section 5.2: Notebooks three_datasets_distr.ipynb and three_datasets_point.ipynb.

- Section 5.3: Notebooks test_distr_bnn.ipynb and test_point_bnn.ipynb.

- Supplemental Section 3.2: Notebook toy_distr_dependent.ipynb.

- Supplemental Sections 5 and 6: Notebooks toy_distr.ipynb and toy_point.ipynb.
