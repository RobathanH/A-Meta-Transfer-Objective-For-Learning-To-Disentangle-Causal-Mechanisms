from typing import Iterable
from itertools import chain
import numpy as np
from tqdm.notebook import trange as tnrange

import torch
import torch.nn as nn

from .augmented_binary_models import *
from cs330.data_generators.multivariate_categorical import MultiCategorical
from cs330.data_generators.causal_graph import CausalGraph


'''
Performs meta-learning on a given causal graph and binary subset structural model to determine 
the causal relationship between two causal variables
'''
class AugmentedBinaryModelTrainer:
    '''
    Args:
        inner_lr (float)
        outer_lr (float)
        transfer_episode_count (int)
        transfer_episode_gradient_steps (int)
        transfer_episode_batch_size (int)
    '''
    def __init__(self, 
        data_generator: MultiCategorical,
        structural_model: BinarySubsetStructuralModel,
        inner_lr: float = 1e-1,
        outer_lr: float = 1e-2,
        transfer_episode_count: int = 500,
        transfer_episode_gradient_steps: int = 20,
        transfer_episode_batch_size: int = 50,
        pretrain_episode_batch_size: int = 500
    ) -> None:

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.transfer_episode_count = transfer_episode_count
        self.transfer_episode_gradient_steps = transfer_episode_gradient_steps
        self.transfer_episode_batch_size = transfer_episode_batch_size
        self.pretrain_episode_batch_size = pretrain_episode_batch_size

        # Ground truth causal structure and data generator
        self.data_generator = data_generator
        self.pretrain_samples = self.data_generator.sample(self.pretrain_episode_batch_size)

        # Binary subset structure model
        self.structural_model = structural_model
        self.hypothesis_opt = torch.optim.SGD(self.structural_model.hypothesis_parameters(), lr=self.inner_lr)
        self.structural_opt = torch.optim.RMSprop(self.structural_model.structure_parameters(), lr=self.outer_lr)

        # Logging buffers
        self.structure_likelihoods = torch.zeros((self.transfer_episode_count, self.structural_model.hypothesis_count()))

    def reset(self) -> None:
        self.structural_model.reset_structure_parameters()

    # --- Training Loops ---

    '''
    Train structure parameters over multiple transfer episodes
    '''
    def train_structure(self) -> None:
        for transfer_episode in tnrange(self.transfer_episode_count, leave=False):
            self.train_transfer_episode()
            self.structure_likelihoods[transfer_episode, :] = self.structural_model.structure_likelihoods()

    '''
    Train on a single transfer episode 
    '''
    def train_transfer_episode(self) -> None:
        # Reset hypothesis models to pretrain distribution state
        self.structural_model.set_maximum_likelihood(self.pretrain_samples)

        # Adjust root variable distributions in causal graph, initializing a
        # new transfer distribution
        self.data_generator.reset_root_distributions()
        transfer_samples = self.data_generator.sample(self.transfer_episode_batch_size)

        # Zero all model gradients to be safe
        self.structural_model.zero_grad()

        # Accumulate log likelihood over training for structural model
        structural_model_regret = torch.tensor(0.)
        for grad_step in range(self.transfer_episode_gradient_steps):
            # Add to meta-loss
            structural_model_regret += -torch.mean(self.structural_model(transfer_samples))
            
            # Perform inner-loop gradient step
            self.hypothesis_opt.zero_grad()
            total_hypothesis_model_loss = torch.tensor(0.)
            for hypothesis_model in self.structural_model.hypothesis_models():
                total_hypothesis_model_loss += -torch.mean(hypothesis_model(transfer_samples))
            total_hypothesis_model_loss.backward()
            self.hypothesis_opt.step()

        # Perform outer-loop gradient step
        self.structural_opt.zero_grad()
        structural_model_regret.backward()
        self.structural_opt.step()