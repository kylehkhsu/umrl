import numpy as np
import torch
import ipdb
from mixture.models import GaussianMixture
from torch.distributions.multivariate_normal import MultivariateNormal

def _estimate_gaussian_parameters(X, resp, reg_covar=1e-6):
    n_trajectories, n_time_steps, n_features = X.shape

    nk = resp.sum(dim=0) + 1e-7

    n_components = resp.shape[1]

    means = torch.Tensor(n_components, n_time_steps, n_features).cuda()

    for k in range(n_components):
        for j in range(n_time_steps):
            means[k, j] = torch.sum(resp[:, k].unsqueeze(1) * X[:, j], dim=0) / nk[k]

    covariances = torch.Tensor(n_components, n_time_steps, n_features, n_features).cuda()

    for k in range(n_components):
        for j in range(n_time_steps):
            diff = X[:, j] - means[k, j]
            covariances[k, j] = torch.mm(resp[:, k] * diff.t(), diff) / nk[k]
            covariances[k, j].view(-1)[::n_features + 1] += reg_covar

    return nk, means, covariances

def _e_step(X, means, covariances):
    n_trajectories, n_time_steps, n_features = X.shape
    n_components = means.shape[0]

    # log probs
    log_prob = torch.Tensor(n_trajectories, n_time_steps, n_components).cuda()
    for k in range(n_components):
        for j in range(n_time_steps):
            m = MultivariateNormal(loc=means[k, j], covariance_matrix=covariances[k, j])
            log_prob[:, j, k] = m.log_prob(X[:, j])

    # modified


    ipdb.set_trace()
    x=1





def gmm(trajectories):
    n_trajectories, n_time_steps, n_features = trajectories.shape
    n_components = 20
    X = torch.Tensor(trajectories).cuda()

    # random initialization
    resp = np.random.rand(n_trajectories, n_components)
    resp /= resp.sum(axis=1)[:, None]
    resp = torch.Tensor(resp).cuda()

    nk, means, covariances = _estimate_gaussian_parameters(X, resp)

    max_iter = 100
    lower_bound = torch.tensor(-np.infty)
    for n_iter in range(1, max_iter + 1):
        prev_lower_bound = lower_bound
        log_prob_norm, log_resp = _e_step(X, means, covariances)



## calculate mean


def check(trajectories):
    x = trajectories.reshape([-1, 2])
    model = GaussianMixture(n_components=5)
    model.fit(x)


if __name__ == '__main__':
    trajectories = np.load('./mixture/data_itr20.pkl')
    gmm(trajectories)
    # check(trajectories)