import hydra
import numpy as np
import torch


class DummyAgent:
    def __init__(self, name, obs_type, obs_shape, action_dim, device, batch_size, nstep):
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.obs_type = obs_type
        
        self.train()
        
        
    def train(self, training=True):
        self.training = training
        
    def update(self, batch, step):
        return  dict()
    
  