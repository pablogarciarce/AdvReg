import numpy as np
import torch
from torch.optim import SGD, Adam
from joblib import Parallel, delayed
import time



def kl_div(mu_n, lam_n, sigma2, x, x_adv):
    sigma2_A = x_adv.T @ torch.inverse(lam_n) @ x_adv + sigma2
    mu_A = x_adv.T @ mu_n
    sigma2_D = x.T @ torch.inverse(lam_n) @ x + sigma2
    mu_D = x.T @ mu_n
    kl = 0.5 * (torch.log(sigma2_A / sigma2_D) + (sigma2_D + (mu_D - mu_A)**2) / sigma2_A - 1)
    return kl

def kl_to_appd(mu_n, lam_n, sigma2, x_adv, mu_D, sigma2_D):
    sigma2_A = x_adv.T @ torch.inverse(lam_n) @ x_adv + sigma2
    mu_A = x_adv.T @ mu_n
    kl = 0.5 * (torch.log(sigma2_A / sigma2_D) + (sigma2_D + (mu_D - mu_A)**2) / sigma2_A - 1)
    return kl

# kl maximization to find adversarial attacked to a trained model
def kl_maximization(model, x, lr=0.01, n_iter=100, epsilon=.3):
    x_adv_values = []
    kl_values = []
    
    mu_n = model.mu
    lam_n = model.lam
    sigma2 = model.sigma2
    x_adv = (x + torch.randn_like(x) * 0.001).clone().detach().requires_grad_(True)  # add some noise to the input so kl is not zero
    optimizer = SGD([x_adv], lr=lr)
    for _ in range(n_iter):
        x_adv.requires_grad = True
        optimizer.zero_grad()
        kl = - kl_div(mu_n, lam_n, sigma2, x, x_adv)  # maximum disruption problem
        kl.backward()
        optimizer.step()
        x_adv.grad.zero_()
        
        with torch.no_grad():
            if torch.norm(x_adv - x, p=2) > epsilon:
                x_adv = x + epsilon * (x_adv - x) / torch.norm(x_adv - x, p=2)
            
        x_adv_values.append(x_adv.clone().detach().numpy())
        kl_values.append(-kl.detach().item())

    return x_adv.detach(), x_adv_values, kl_values 


# Function pi(y | x, gamma)
def pi(y, x, gamma):
    return torch.distributions.normal.Normal(x.T @ gamma[0], gamma[1]).log_prob(y).exp()

# Gradient of pi(y | x, gamma) with respect to x
# pi(y | x, gamma) is Normal(x.T @ beta, sigma2) with beta = gamma[0] and sigma2 = gamma[1]
def grad_pi(y, x, gamma): 
    distr = torch.distributions.normal.Normal(x.T @ gamma[0], gamma[1])
    prob = distr.log_prob(y).exp()
    grad = (y - x.T @ gamma[0]) / gamma[1] * prob * gamma[0]
    return grad

# g_{x, M}(y)
def g_x_M(y, x, gamma_samples): 
    # betas and sigmas shape: (D, M) and (1, M)
    grad_pi_vals = grad_pi(y, x, gamma_samples) 
    pi_vals = pi(y, x, gamma_samples) 
    numerator = torch.mean(grad_pi_vals, dim=1, keepdim=True)  # Promedio sobre M (segunda dimensión)
    denominator = torch.mean(pi_vals, dim=1)  # Promedio sobre M (segunda dimensión)
    return numerator / denominator  # not - since max disruption problem

# Compute Δg_{x, l}(y)
def delta_g_x_l(y, x, l, model, M_sequence):
    M_l = M_sequence[l]
    M_l_minus_1 = M_sequence[l-1] if l > 0 else 0
    gamma_samples_l = model.sample_posterior_distribution(M_l)
    
    # using the same samples for both terms in the difference
    gamma_samples_l_minus_1_a = [gamma_samples_l[0][:, :M_l_minus_1], gamma_samples_l[1][:, :M_l_minus_1]]
    gamma_samples_l_minus_1_b = [gamma_samples_l[0][:, M_l_minus_1:], gamma_samples_l[1][:, M_l_minus_1:]]

    g_l = g_x_M(y, x, gamma_samples_l)
    g_l_minus_1_a = g_x_M(y, x, gamma_samples_l_minus_1_a) if l > 0 else 0
    g_l_minus_1_b = g_x_M(y, x, gamma_samples_l_minus_1_b) if l > 0 else 0
    return g_l - (g_l_minus_1_a + g_l_minus_1_b) / 2

# Estimate the gradient using MLMC in parallel with joblib
def mlmc_gradient_estimator(y, x, R, model, M0=1, tau=1., n_jobs=50):
    # Define sequence M_l
    M_sequence = [M0*2**l for l in range(100)]

    # Define weights ω_l
    omega = [2**(-tau * l) for l in range(len(M_sequence))]
    omega = np.array(omega)
    omega /= omega.sum()  # Normalize

    l_indices = np.random.choice(len(M_sequence), size=R, p=omega)
    
    estimates = [delta_g_x_l(y, x, l, model, M_sequence) / omega[l] for l in l_indices]  
    
    return sum(estimates) / R

# Attack function to use the gradient estimator for maximum disruption
def mlmc_attack(model, x, appd=None, lr=0.01, n_iter=1000, epsilon=.1, R=100, early_stopping_patience=10, verbose=True,
                optimizer='Adam'):
    """
    Function to perform the attack using the MLMC gradient estimator.
    :param appd: Attacker predictive posterior distribution to approximate. If None -> Maximum disruption attack.
    """
    x_adv_values = []
    patience = 0
    x_adv = (x + torch.randn_like(x) * 0.0001).clone().requires_grad_(True)  # add some noise to the input
    if optimizer == 'SGD':
        optimizer = SGD([x_adv], lr=lr, momentum=.5, dampening=.95)
    elif optimizer == 'Adam':
        optimizer = Adam([x_adv], lr=lr)
    else: 
        raise ValueError('Optimizer not recognized')

    for it in range(n_iter):
        x_adv.requires_grad = True
        optimizer.zero_grad()
        if appd is None:
            y = model.sample_predictive_distribution(x, num_samples=1)
            x_adv.grad = mlmc_gradient_estimator(y, x_adv, R, model)
        else:
            x_adv.grad = -mlmc_gradient_estimator(appd.sample(), x_adv, R, model)
        optimizer.step()
        x_adv.grad.zero_()
        
        with torch.no_grad():
            if torch.norm(x_adv - x, p=2) > epsilon:
                x_adv = x + epsilon * (x_adv - x) / torch.norm(x_adv - x, p=2)
            
        x_adv_values.append(x_adv.clone().detach().numpy())

        if (it > 2 and np.linalg.norm(x_adv_values[-1] - x_adv_values[-2]) < 1e-4):
            patience += 1
            if patience >= early_stopping_patience:
                if verbose: 
                    print(f'Early stopping at iteration {it}')
                break
        else:
            patience = 0

    return x_adv.detach(), x_adv_values

# Attack function to use the gradient estimator for maximum disruption
def fgsm_attack(model, x, appd=None, lr=0.01, n_iter=1000, epsilon=.1, R=100, early_stopping_patience=10):
    """
    Function to perform the FGSM attack using the MLMC gradient estimator.
    :param appd: Attacker predictive posterior distribution to approximate. If None -> Maximum disruption attack.
    """
    x_adv = (x + torch.randn_like(x) * 0.00001).clone().requires_grad_(True)
    optimizer = SGD([x_adv], lr=lr)

    x_adv.requires_grad = True
    optimizer.zero_grad()
    if appd is None:
        y = model.sample_predictive_distribution(x, num_samples=1)
    else:
        y = appd.sample()
    grad = mlmc_gradient_estimator(y, x_adv, R, model)
    if appd is None:
        x_adv.grad = grad.sign()
    else:
        x_adv.grad = - grad.sign()  # If appd is not None, we want to minimize the loss
        
    optimizer.step()
    x_adv.grad.zero_()
    
    with torch.no_grad():
        x_adv = x + epsilon * (x_adv - x) / torch.norm(x_adv - x, p=2)
        
    return x_adv.detach()