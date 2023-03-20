import torch
import numpy as np
import ot

from sklearn import preprocessing


def cosine_distance(x, y):
    return ot.utils.dist(x, y, metric="cosine")


def euclidean_distance(x, y):
    return ot.utils.dist(x, y, metric="euclidean") / x.shape[1]


COSTS = {"euclidean": euclidean_distance, "cosine": cosine_distance}


class Cost:
    def __init__(self, cost_function, normalize="none"):
        self.cost_function = cost_function
        self.normalize = normalize

    def compute(self, x, y):
        c = COSTS[self.cost_function](x, y)
        if self.normalize == "max":
            c = c / c.max()
        return c


class EllipticalCost:
    def __init__(self, expert_features, reg=1e-8):
        self.reg = reg
        self.inv_cov = self.inverse_cov(expert_features)

    def inverse_cov(self, feat):
        # (batch, phi_dim)
        phi = np.concatenate(feat, axis=0)
        feat_dim = phi.shape[-1]
        cov = np.mean(phi[:, :, np.newaxis] @ phi[:, np.newaxis, :], axis=0)
        cov += self.reg * np.eye(feat_dim)
        # For now computing psuedo inv. Maybe need to do regularization
        return np.linalg.inv(cov)

    def compute(self, x, y):
        return x @ self.inv_cov @ y.T


# Solvers
def sinkhorn(x, y, cost_matrix, epsilon=0.01, niter=100):
    mu_x = np.ones(x.shape[0]) * (1 / x.shape[0])
    mu_y = np.ones(y.shape[0]) * (1 / y.shape[0])
    return ot.sinkhorn(mu_x, mu_y, cost_matrix, epsilon, numItermax=niter)


def emd(x, y, cost_matrix, niter=100000):
    mu_x = np.ones(x.shape[0]) * (1 / x.shape[0])
    mu_y = np.ones(y.shape[0]) * (1 / y.shape[0])
    return ot.emd(mu_x, mu_y, cost_matrix, numItermax=niter)


# TODO: softdtw.....

SOLVERS = {"sinkhorn": sinkhorn, "emd": emd}


class Solver:
    def __init__(self, solver_function, epsilon=None):
        self.solver_function = solver_function
        self.epsilon = epsilon

    def compute(self, x, y, cost_matrix):
        if self.solver_function == "sinkhorn":
            return SOLVERS[self.solver_function](
                x, y, cost_matrix, epsilon=self.epsilon
            )
        return SOLVERS[self.solver_function](x, y, cost_matrix)


# Aggregators
def topk(rew_traj_list, return_list, topk=3):
    topk_closest_demo_idx = torch.topk(torch.tensor(return_list), topk).indices
    topk_returns = [
        traj
        for i, traj in enumerate(rew_traj_list)
        if i in topk_closest_demo_idx.tolist()
    ]
    return np.mean(topk_returns, 0)


def argmax(rew_traj_list, return_list, topk=None):
    closest_demo_idx = np.argmax(return_list)
    return rew_traj_list[closest_demo_idx]


def mean(rew_traj_list, return_list, topk=None):
    return np.mean(rew_traj_list, 0)


AGGREGATORS = {"topk": topk, "argmax": argmax, "mean": mean}


class Aggregator:
    def __init__(self, aggregator_function, topk):
        self.aggregator_function = aggregator_function
        self.topk = topk

    def compute(self, rew_traj_list, return_list):
        return AGGREGATORS[self.aggregator_function](
            rew_traj_list, return_list, self.topk
        )


# Preprocessors
class NoScaler:
    def __init__(self):
        pass

    def fit(self, observations):
        pass

    def transform(self, trajectory):
        return trajectory


PREPROCESSORS = {
    "standardScaler": preprocessing.StandardScaler(),
    "noScaler": NoScaler(),
}


class Preprocessor:
    def __init__(
        self,
        preprocessor_function="standardScaler",
        partial_updates=True,
        update_preprocessor_every_episode=25,
    ):
        self.preprocessor = PREPROCESSORS[preprocessor_function]
        self.update_freq = update_preprocessor_every_episode
        self.partial_updates = partial_updates

    def fit(self, observations, episode):
        if episode % self.update_freq == 0:
            if self.partial_updates:
                self.preprocessor.partial_fit(observations)
            else:
                self.preprocessor.fit(observations)

    def transform(self, observations):
        return self.preprocessor.transform(observations)


def linear_scaling(rewards, alpha=10.0):
    return -alpha * rewards


def poly_scaling(rewards, alpha=10.0, beta=2.0):
    return -alpha * (rewards**beta)


def exponential_scaling(rewards, alpha=5.0, beta=5.0):
    return alpha * np.exp(-beta * rewards * rewards.shape[0])


reward_scalings = {
    "linear": linear_scaling,
    "exponential": exponential_scaling,
    "quad": poly_scaling,
}
