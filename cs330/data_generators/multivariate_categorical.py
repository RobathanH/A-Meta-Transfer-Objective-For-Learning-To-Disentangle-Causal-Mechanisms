import numpy as np
from typing import Tuple
import torch
import torch.nn as nn

from .causal_graph import CausalGraph

'''
Interface for generating and storing multivariate functional causal models,
in which each node is an N-way categorical variable, and functions between
nodes are parametrized as 2 layer neural networks.
'''

# Constants



class MultiCategorical:
    '''
    Args:
        N (int):                Number of values for each categorical node
        M (int):                Number of nodes in causal structure
        graph (CausalGraph):    Optional graph structure to use. If None,
                                a random graph is created.
    '''
    def __init__(self, N: int, M: int, graph: CausalGraph = None) -> None:
        self.N = N
        self.M = M
        
        if graph is not None:
            self.graph = graph
        else:
            self.graph = CausalGraph(M)
        
        # Number of hidden layers in functional causal edges
        self.H = 4 * max(M, N)

        # Neural networks for each node, implemented manually since we
        # will be changing particular values to alter the distribution
        self.node_functions = [NodeFunction(N, M, self.H) for _ in range(self.M)]

    '''
    Randomizes the distributions for any root variables
    '''
    def reset_root_distributions(self):
        for i in range(self.M):
            if self.graph.is_root(i):
                self.node_functions[i].reset()

    '''
    Randomizes all distributions, but keeps causal graph structure
    '''
    def reset_all_distributions(self):
        for i in range(self.M):
            self.node_functions[i].reset()

    '''
    Performs ancestral sampling to produce a batch of samples
    Args:
        batch_size (int):   Number of samples to create
    Returns:
        (torch.Tensor):     Batch data tensor of shape (batch_size, M, N),
                            where the final axis contains 1-hot representations
                            of node values
    '''
    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            node_vals = torch.zeros(batch_size, self.M * self.N).type(torch.FloatTensor)
            for i in range(self.M):
                mask = torch.as_tensor(np.repeat(self.graph.B[i], self.N)).reshape(1, -1).type(torch.FloatTensor)
                node_vals[:, self.N * i : self.N * (i + 1)] = self.node_functions[i](node_vals * mask).type(torch.FloatTensor)
            node_vals = node_vals.reshape(-1, self.M, self.N)
        return node_vals

    '''
    Approximate ground truth correlation values between two nodes in the graph.
    Args:
        i (int):        index of node for which marginal probability will be approximated
        j (int):        index of node for which conditional probability will be approximated
    Returns:
        (np.array): marginal probability for node j (shape = (N,))
        (np.array): conditional probability for node i given j (shape = (N, N))
    '''
    def compute_correlation(self, i: int, j: int) -> Tuple[np.array, np.array]:
        TOTAL_SAMPLES = 1024
        
        occurrences = np.zeros((self.N, self.N))
        samples = self.sample(TOTAL_SAMPLES)
        sample_occurrences = samples[:, i, :].reshape(-1, self.N, 1) * samples[:, j, :].reshape(-1, 1, self.N)
        occurrences += torch.sum(sample_occurrences, dim=0).numpy()

        P_j = np.sum(occurrences, axis=1) / np.sum(occurrences)
        P_i_j = occurrences / np.sum(occurrences, axis=1, keepdims=True)

        return P_j, P_i_j


class NodeFunction(nn.Module):
    '''
    Args:
        N (int):                    Number of possible values for a single node
        M (int):                    Number of nodes in causal structure
        H (int):                    Size of hidden layer
        bias_init_bound (float):    Bias initialization is uniform below this magnitude
    '''
    def __init__(self, N: int, M: int, H: int, bias_init_bound: float = 0.2) -> None:
        super(NodeFunction, self).__init__()
        
        self.N = N
        self.bias_init_bound = bias_init_bound
        
        self.layer1 = nn.Linear(N * M, H)
        self.layer2 = nn.Linear(H, N)

        # Remove grad
        for p in self.parameters():
            p.requires_grad = False

        # Random initialization
        self.reset()

    '''
    Reinitialize this distribution
    '''
    def reset(self) -> None:
        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.uniform_(self.layer1.bias, -self.bias_init_bound, self.bias_init_bound)
        nn.init.uniform_(self.layer2.bias, -self.bias_init_bound, self.bias_init_bound)

    '''
    Sample this functional edge.
    Args:
        Expects as input a batch of concatenations of each categorical variable's one-hot
        representation, masked to only include the node's valid parents.
        Shape = (batch_size, M * N)
    Returns:
        Samples the final categorical distribution and outputs the resulting
        value as a one_hot tensor
        Shape = (batch_size, N)
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        inds = torch.distributions.Categorical(logits=x).sample()
        vals = nn.functional.one_hot(inds, num_classes=self.N).type(torch.FloatTensor)
        return vals