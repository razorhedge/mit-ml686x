"""Mixture model using EM"""
from operator import mul
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    from scipy.stats import multivariate_normal
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var , p = mixture
    K, _ = mu.shape
    post = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            post[i,j] = p[j] * multivariate_normal.pdf(X[i], mu[j], var[j])
    post = post / post.sum(axis=0)
    return post, np.log(post).sum()
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    n_hat = post.sum(axis=0)
    p_hat = n_hat / n
    mu_hat = np.zeros((K,d))
    sigma_hat = np.zeros(K)
    for j in range(K):
        mu_hat[j,:] = (X*post[:, j, None]).sum(axis=0)/ n_hat[j]
        sse = ((mu_hat[j]-X)**2).sum(axis=1) @ post[:, j]
        sigma_hat[j] = sse / (d * n_hat[j])
    mixture = GaussianMixture(mu_hat, sigma_hat, p_hat)
    return mixture
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """ Runs the mixture model
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    l_old = None
    l_new = None
    while (l_old is None or np.abs(l_new - l_old) >= 1e-6*np.abs(l_new)):
        l_old = l_new
        post, l_new = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, l_new
    raise NotImplementedError
