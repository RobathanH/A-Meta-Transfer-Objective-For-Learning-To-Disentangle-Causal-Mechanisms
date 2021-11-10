import numpy as np
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

'''
Shared format for generating and storing overall causal graphs.
These Causal Graphs contain connection direction information without
any additional assumptions on the functional form of those connections.
To ensure causal graph contains no cycles, node indices within this object
are ordered such that node i can only depend on nodes with index j < i.
This may require reordering node indices when sending them to a model.
'''

# TODO: 
# - Save and load specific graphs
# - Generate short IDs to reference specific graphs


# Constants

# When sampling a random causal graph, this is the maximum bernoulli probability for
# any particular edge to be present
MAX_PARENT_PROBABILITY = 0.5 

# When sampling a random causal graph, the bernoulli probability for each edge
# to be present will be set such that each node will have a constant number of
# parents on average (p = exp_parent_count / possible_parents).
# These probabilities are bounded by MAX_PARENT_PROBABILITY
EXPECTED_PARENT_COUNT = 5 # When sampling a random 


class CausalGraph:
    '''
    Args:
        M (int):                Number of nodes in the causal graph
    '''
    def __init__(self, M: int) -> None:
        # Using Notation from original paper

        # Number of nodes in causal graph
        self.M = M

        # Connection matrix for graph connections
        # B[i, j] = 1 if node j causes node i, else 0
        self.B = np.zeros((M, M))

        # Generate random graph
        self.reset()

    '''
    Sets this causal graph to a random graph
    '''
    def reset(self) -> None:
        allowed_edges = np.tril(np.ones((self.M, self.M)), -1)
        unclipped_probs = allowed_edges * EXPECTED_PARENT_COUNT * np.mean(allowed_edges, axis=1, keepdims=True)
        clipped_probs = np.minimum(unclipped_probs, MAX_PARENT_PROBABILITY)
        self.B = np.random.binomial(1, clipped_probs)

    '''
    Visualize graph structure with networkx
    '''
    def visualize(self):
        G = nx.from_numpy_matrix(self.B.T, create_using=nx.DiGraph)
        return nx.draw(G, with_labels=True, pos=graphviz_layout(G, prog='dot'))

    '''
    Checks if a given variable index is a root variable in this graph
    Args:
        i (int):    index to check
    Returns:
        (bool):     True if the given index corresponds to a root variable
    '''
    def is_root(self, i: int) -> bool:
        return np.sum(self.B[i]) == 0
