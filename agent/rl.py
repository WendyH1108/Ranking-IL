import utils
from .agent import Agent


class RLAgent(Agent):
    def __init__(self, name, task, device, algo):
        super().__init__(name, task, device, algo)

    def __repr__(self):
        return "reinforcement"

    def update(self, replay_iter, step):

        metrics = dict()
        #TODO: make general for other algos
        if step % self.policy.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        metrics.update(self.policy.update(batch, step))

        return metrics
