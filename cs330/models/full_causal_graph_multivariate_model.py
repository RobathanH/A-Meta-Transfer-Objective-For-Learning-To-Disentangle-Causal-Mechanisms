from typing import List
import numpy as np
import itertools
from collections import deque

import torch
import torch.nn as nn

from .causal_parent_multivariate_model import *


'''
Direct adaptation of binary transfer objective to multiple causal graphs, training and comparing performance for
each possible acyclical causal graph.
'''
class AllHypotheses_FullCausalGraphMultivariateModel(CausalParentMultivariateModel):
    # hypothesis_sample_count is unused for this structural model
    def __init__(self, N: int, M: int, hypothesis_sample_count: int) -> None:
        super(AllHypotheses_FullCausalGraphMultivariateModel, self).__init__(N, M, hypothesis_sample_count)

        # Constants
        # Repeat each hypothesis multiple times for stability
        self.HYPOTHESIS_REPEATS = 3

        # Compute all possible directed acyclical graphs, to store and train a hypothesis model for each
        self.all_hypotheses = torch.from_numpy(all_acyclic_causal_graphs(self.M)).detach().type(torch.FloatTensor)

        # Hypotheses will be referenced internally via index, so we create
        # a dict to quickly find hypothesis index from tensor
        self.hypothesis_index_dict = {h.numpy().tobytes(): i for i, h in enumerate(self.all_hypotheses)}

        # Save pretrained state for each hypothesis
        self.pretrained_state = [None for _ in range(len(self.all_hypotheses))]

        # Save original untrained node_function state to reset between pretraining for each hypothesis
        self.init_state = self.node_functions.state_dict()

        # Structure parameters (trained in outer loop) are an array of pre-softmax likelihoods for each
        # possible full-graph hypothesis
        self.structure_weights = nn.Parameter(torch.zeros(len(self.all_hypotheses)))


    '''
    Pretrain separately for each full-graph hypothesis, since that is what the binary transfer objective does.
    '''
    def pretrain_hypotheses(self) -> torch.Tensor:
        return self.all_hypotheses

    '''
    Save pretrained state for each hypothesis, pretrained one at a time.
    Also resets node functions to initial untrained state.
    '''
    def save_pretrained(self, pretrain_hypothesis: torch.Tensor) -> None:
        self.pretrained_state[self.hypothesis_index_dict[pretrain_hypothesis.numpy().tobytes()]] = self.node_functions.state_dict()
        self.node_functions.load_state_dict(self.init_state)

    '''
    Each outer loop step must train on each hypothesis, just as the binary transfer objective does.
    '''
    def sample_hypotheses(self) -> torch.Tensor:
        return self.all_hypotheses.repeat(self.HYPOTHESIS_REPEATS, 1, 1)

    '''
    Load pretrained state for each hypothesis, ahead of training it on a transfer distribution
    '''
    def load_pretrained(self, hypothesis_to_test: torch.Tensor) -> None:
        correct_pretrained_state = self.pretrained_state[self.hypothesis_index_dict[hypothesis_to_test.numpy().tobytes()]]
        if correct_pretrained_state is None:
            raise ValueError(f"Pretrained state for hypothesis index {self.hypothesis_index_dict[hypothesis_to_test.numpy().tobytes()]} was never loaded.")

        self.node_functions.load_state_dict(correct_pretrained_state)


    '''
    Compute structure gradients based on automatic differentiation of explicit regret, which is
    the negative log of the expected value of online likelihood over each possible hypothesis.
    Since hypothesis_online_log_likelihood stores log likelihoods separately for each predicted node, those likelihoods
    must be aggregated to get log likelihood for each full-graph hypothesis
    '''
    def compute_structure_gradients(self, hypothesis_list: torch.FloatTensor, hypothesis_online_log_likelihoods: torch.FloatTensor) -> None:
        hypothesis_structure_log_likelihoods = self.structure_weights.repeat(self.HYPOTHESIS_REPEATS) - torch.logsumexp(self.structure_weights, dim=0)
        hypothesis_online_training_log_likelihoods = torch.sum(hypothesis_online_log_likelihoods, dim=1)

        regret = -torch.logsumexp(hypothesis_structure_log_likelihoods + hypothesis_online_training_log_likelihoods, dim=0)

        regret.backward()

    '''
    Reset structure parameters
    '''
    def reset_structure_parameters(self) -> None:
        self.structure_weights.data = torch.zeros(len(self.structure_weights))

    # --- Logging and info methods ---

    '''
    Return the current predicted most-likely full-graph hypothesis
    '''
    def predicted_graph(self) -> torch.FloatTensor:
        return self.all_hypotheses[torch.argmax(self.structure_weights)]

    '''
    Return current structure likelihoods (chance that each node is parent of target node)
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        return torch.softmax(self.structure_weights, dim=0).detach()

    '''
    Returns names identifying each value in structure_likelihoods().
    Args:
        index:  Tuple of integers indexing a particular value in structure_likelihoods().
                For this structural model, that means just one integer index.
    Returns:
        (str):  A short string identifying the meaning of that structure parameter index.
    '''
    def structure_param_name(self, *index) -> str:
        index = index[0]
        hypothesis = self.all_hypotheses[index]
        return f"{[round(x) for x in hypothesis.flatten().tolist()]}"


'''
Version of the CausalParentMultivariateModel which doesn't make the 
independent target node assumption, and thus samples from the space
of all possible full causal graphs.
Structural params for each node pair are now a softmax between three options:
    A -> B, B -> A, and no-edge
'''
class AllHypotheses_IndependentEdgeFullCausalGraphMultivariateModel(CausalParentMultivariateModel):
    def __init__(self, N: int, M: int, hypothesis_sample_count: int) -> None:
        super(AllHypotheses_IndependentEdgeFullCausalGraphMultivariateModel, self).__init__(N, M, hypothesis_sample_count)

        # Constants
        # Repeat each hypothesis multiple times for stability
        self.HYPOTHESIS_REPEATS = 3

        # Compute all possible directed acyclical graphs, to store and train a hypothesis model for each
        self.all_hypotheses = torch.from_numpy(all_acyclic_causal_graphs(self.M)).detach().type(torch.FloatTensor)

        # Hypotheses will be referenced internally via index, so we create
        # a dict to quickly find hypothesis index from tensor
        self.hypothesis_index_dict = {h.numpy().tobytes(): i for i, h in enumerate(self.all_hypotheses)}

        # Save pretrained state for each hypothesis
        self.pretrained_state = [None for _ in range(len(self.all_hypotheses))]

        # Save original untrained node_function state to reset between pretraining for each hypothesis
        self.init_state = self.node_functions.state_dict()

        # Base structure parameters for optimization
        self.structure_weights_edge_base = nn.Parameter(torch.zeros(self.M, self.M))
        self.structure_weights_no_edge_base = nn.Parameter(torch.zeros(self.M, self.M))

        self.structure_weights = None

    # Internal helper function which converts base optimizable parameters into structure weights which hide implicit constraints
    # Returned tensor has shape (M, M, 3)
    # [i, j, 0] = P(i <-- j),       [i, j, 1] = P(i --> j),     [i, j, 2] = P(i _|_ j)
    # [i, j, 0] == [j, i, 1]        P(i <-- j) == P(j --> i)
    # [i, j, 2] == [j, i, 2]        P(i _|_ j) == P(j _|_ i)
    def get_structure_weights(self):
        return torch.stack([
            self.structure_weights_edge_base,
            self.structure_weights_edge_base.transpose(0, 1),
            torch.triu(self.structure_weights_no_edge_base, diagonal=1) + torch.tril(self.structure_weights_no_edge_base.transpose(0, 1), diagonal=-1)
        ], dim=2)

    

    '''
    Pretrain separately for each full-graph hypothesis, since that is what the binary transfer objective does.
    '''
    def pretrain_hypotheses(self) -> torch.Tensor:
        return self.all_hypotheses

    '''
    Save pretrained state for each hypothesis, pretrained one at a time.
    Also resets node functions to initial untrained state.
    '''
    def save_pretrained(self, pretrain_hypothesis: torch.Tensor) -> None:
        self.pretrained_state[self.hypothesis_index_dict[pretrain_hypothesis.numpy().tobytes()]] = self.node_functions.state_dict()
        self.node_functions.load_state_dict(self.init_state)

    '''
    Each outer loop step must train on each hypothesis, just as the binary transfer objective does.
    '''
    def sample_hypotheses(self) -> torch.Tensor:
        return self.all_hypotheses.repeat(self.HYPOTHESIS_REPEATS, 1, 1)

    '''
    Load pretrained state for each hypothesis, ahead of training it on a transfer distribution
    '''
    def load_pretrained(self, hypothesis_to_test: torch.Tensor) -> None:
        correct_pretrained_state = self.pretrained_state[self.hypothesis_index_dict[hypothesis_to_test.numpy().tobytes()]]
        if correct_pretrained_state is None:
            raise ValueError(f"Pretrained state for hypothesis index {self.hypothesis_index_dict[hypothesis_to_test.numpy().tobytes()]} was never loaded.")

        self.node_functions.load_state_dict(correct_pretrained_state)


    def compute_structure_gradients(self, hypothesis_list: torch.FloatTensor, hypothesis_online_log_likelihoods: torch.FloatTensor) -> None:
        # Convert hypothesis list into same three channels as get_structure_weights(): forward edge, backward edge, and no edge
        hypothesis_list_forward_edge = hypothesis_list
        hypothesis_list_backward_edge = hypothesis_list.transpose(1, 2)
        hypothesis_list_no_edge = (hypothesis_list_forward_edge + hypothesis_list_backward_edge) == 0
        hypothesis_list_with_channels = torch.stack([
            hypothesis_list_forward_edge,
            hypothesis_list_backward_edge,
            hypothesis_list_no_edge
        ], dim=3)
        hypothesis_list_with_channels_upper_tri = hypothesis_list_with_channels * torch.triu(torch.ones(self.M, self.M), diagonal=1).reshape(1, self.M, self.M, 1)

        # Compute log likelihood for each hypothesis, aggregating over each edge pair (avoiding double-counting by using only upper triangular part)
        structure_log_likelihoods = (self.get_structure_weights() - torch.logsumexp(self.get_structure_weights(), dim=2, keepdim=True)).reshape(1, self.M, self.M, -1)
        hypothesis_log_likelihoods = torch.sum(hypothesis_list_with_channels_upper_tri * structure_log_likelihoods, dim=(1, 2, 3))

        # Compute online training log likelihood for each full-graph hypothesis transfer training
        hypothesis_online_training_log_likelihoods = torch.sum(hypothesis_online_log_likelihoods, dim=1)

        # Use structure-based hypothesis prior likelihoods to compute regret as expectation of log of online training likelihood
        regret = -torch.logsumexp(hypothesis_log_likelihoods + hypothesis_online_training_log_likelihoods, dim=0)
        regret.backward()


    '''
    Reset structure parameters
    '''
    def reset_structure_parameters(self) -> None:
        self.structure_weights_edge_base.data = torch.zeros(self.M, self.M)
        self.structure_weights_no_edge_base.data = torch.zeros(self.M, self.M)

    '''
    Return list of structure parameters (for outer-loop training)
    '''
    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.structure_weights_edge_base, self.structure_weights_no_edge_base]

    # --- Logging and info methods ---

    '''
    Return the current predicted most-likely full-graph hypothesis
    '''
    def predicted_graph(self) -> torch.FloatTensor:
        structure_weights = self.get_structure_weights()
        return torch.argmax(structure_weights, dim=2) == 0

    '''
    Return current structure likelihoods (chance that each node is parent of target node)
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        return torch.softmax(self.get_structure_weights(), dim=2)

    '''
    Returns names identifying each value in structure_likelihoods().
    Args:
        index:  Tuple of integers indexing a particular value in structure_likelihoods().
    Returns:
        (str):  A short string identifying the meaning of that structure parameter index.
    '''
    def structure_param_name(self, *index) -> str:
        A, B, channel = index
        if channel == 0:
            return f"Node {A} <-- Node {B}"
        if channel == 1:
            return f"Node {A} --> Node {B}"
        if channel == 2:
            return f"Node {A} _|_ Node {B}"