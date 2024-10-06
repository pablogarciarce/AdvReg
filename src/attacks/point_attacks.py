import numpy as np
import torch
from torch.optim import SGD

from src.utils import id



def reparametrization_trick(x_adv, model, G, samples_per_iteration, func):
    # Sample from the model's predictive distribution keeping the gradient flow by using rsample
    y_samples = model.get_predictive_distribution(x_adv.unsqueeze(1)).rsample((samples_per_iteration,))
    
    # Using autograd to compute the gradients
    f_values = func(x_adv, y_samples)
    loss = ((f_values - G) ** 2).mean()
    loss.backward()
    
    return x_adv.grad, f_values.mean(), loss

def attack(x_clean, model, G, samples_per_iteration=100, learning_rate=1e-3, num_iterations=1000, epsilon=.1, func=id, early_stopping_patience=10):
    x_0 = x_clean.clone().detach()
    x_adv = (x_clean + torch.randn_like(x_clean) * 0.0002).clone().detach().requires_grad_(True)
    x_adv_values = []
    loss_values = []
    func_values = []
    early_stopping_it = 0

    optimizer = SGD([x_adv], lr=learning_rate, momentum=0.9)

    for _ in range(num_iterations):
        x_adv.requires_grad = True
        old_x_adv = x_adv.clone().detach()

        _, f_mean, loss = reparametrization_trick(x_adv, model, G, samples_per_iteration, func=func)

        optimizer.step()
        x_adv.grad.zero_()
        
        with torch.no_grad():
            if torch.norm(x_adv - x_0, p=2) > epsilon:
                x_adv = x_0 + epsilon * (x_adv - x_0) / torch.norm(x_adv - x_0, p=2)

        # Early stopping
        if torch.norm(x_adv - old_x_adv, p=2) < 1e-5:
            early_stopping_it += 1
            if early_stopping_it > early_stopping_patience:
                break
        else:
            early_stopping_it = 0            
        x_adv_values.append(x_adv.clone().detach().numpy())
        loss_values.append(loss.item())
        func_values.append(f_mean.item())

    return x_adv_values, loss_values, func_values

def true_gradient_mean(x_adv, model, G):
    # The true gradient of the mean of the predictive distribution
    beta_dot_x = model.mu @ x_adv
    return 2 * (beta_dot_x - G) * model.mu

def attack_true_grad(x_clean, model, G, samples_per_iteration=1000, learning_rate=1e-1, num_iterations=1000, epsilon=.1, func=id, early_stopping_patience=10):
    x_0 = x_clean.clone().detach()
    x_adv = x_clean.clone().detach().requires_grad_(True)
    x_adv_values = []
    # early_stopping_it = 0

    optimizer = torch.optim.SGD([x_adv], lr=learning_rate)

    for _ in range(num_iterations):
        x_adv.requires_grad = True
        # old_x_adv = x_adv.clone().detach()

        x_adv.grad = true_gradient_mean(x_adv, model, G)
        optimizer.step()
        x_adv.grad.zero_()
        # print(torch.norm((x_adv - x_0) / torch.norm(x_adv - x_0, p=2) - model.mu / torch.norm(model.mu, p=2)))
        with torch.no_grad():
            if torch.norm(x_adv - x_0, p=2) > epsilon:
                x_adv = x_0 + epsilon * (x_adv - x_0) / torch.norm(x_adv - x_0, p=2)

        # Early stopping
        # if torch.norm(x_adv - old_x_adv, p=2) < 1e-5:
        #     early_stopping_it += 1
        #     if early_stopping_it > early_stopping_patience:
        #         break
        # else:
        #     early_stopping_it = 0            
        # print(torch.norm((x_adv - x_0) / torch.norm(x_adv - x_0, p=2) - model.mu / torch.norm(model.mu, p=2)))
        x_adv_values.append(x_adv.clone().detach().numpy())
    # print(torch.norm((x_adv - x_0) / torch.norm(x_adv - x_0, p=2) - model.mu / torch.norm(model.mu, p=2)))
    return x_adv_values

def attack_fgsm(x_clean, model, G, samples_per_iteration=1000, learning_rate=1e-3, num_iterations=1000, epsilon=.1, func=id, early_stopping_patience=10):
    x_0 = x_clean.clone().detach()
    x_adv = (x_clean + torch.randn_like(x_clean) * 0.0002).clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([x_adv], lr=learning_rate)

    x_adv.requires_grad = True

    _, f_mean, loss = reparametrization_trick(x_adv, model, G, samples_per_iteration, func=func)
    x_adv.grad = x_adv.grad.sign()
    optimizer.step()
    x_adv.grad.zero_()
    
    with torch.no_grad():
        x_adv = x_0 + epsilon * (x_adv - x_0) / torch.norm(x_adv - x_0, p=2)

    return x_adv

def det_attack(x_adv, model, y_star, epsilon=.1, verbose=False):
    mu = model.mu.numpy()

    beta_dot_x = np.dot(mu, x_adv)
    beta_norm_squared = np.dot(mu, mu)
    x_adv_det = x_adv + ((y_star - beta_dot_x) / beta_norm_squared) * mu

    if np.linalg.norm(np.dot(y_star - beta_dot_x, mu)) > epsilon * beta_norm_squared: 
        if verbose:
            print('Optimal perturbation is too large')
        delta_x_adv = epsilon * (np.dot(y_star - beta_dot_x, mu) / (np.linalg.norm(y_star - beta_dot_x) * np.linalg.norm(mu)))
        x_adv_det = x_adv + delta_x_adv

    y_adv_det = np.dot(mu, x_adv_det)
    return x_adv_det, y_adv_det
