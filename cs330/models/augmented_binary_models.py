from typing import Iterable, List
from itertools import chain
import numpy as np

import torch
import torch.nn as nn

# Constants
EPS = 1e-9 # to avoid divide-by-zero error in log



'''
Class interface for a model representing a particular possible relationship between
two nodes in a multivariate system
'''
class HypothesisModel:
    '''
    Set all model parameters to maximize sample likelihood.
    Used for pretraining on source distribution before transfer training.
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    '''
    def set_maximum_likelihood(self, samples: torch.FloatTensor) -> None:
        raise NotImplementedError

'''
Model for approximating two-variable relationship as a directed
causal edge, with marginal and conditional probabilities.
Given samples drawn from a distribution, outputs their log likelihood.
'''
class CauseModel(nn.Module, HypothesisModel):
    '''
    Args:
        N (int):        Number of values for each categorical variable
        M (int):        Total number of variables in the causal graph
        node_1 (int):   Graph index for the node assumed to be the cause
        node_2 (int):   Graph index for the node assumed to be the effect
    '''
    def __init__(self, N: int, M: int, node_1: int, node_2: int) -> None:
        nn.Module.__init__(self)

        self.N = N
        self.M = M
        self.node_1 = node_1
        self.node_2 = node_2
        
        # Initialize modules
        self.P_1 = nn.Parameter(torch.zeros(N)) # marginal unnormalized log probability table
        self.P_2_1 = nn.Parameter(torch.zeros(N, N)) # conditional unnormalized log probability table

    '''
    Forward pass computes the log likelihood of the given set of variable samples
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    Returns:
        (torch.FloatTensor):            Overall log likelihood of the given samples under
                                        this model's current parameters.
    '''
    def forward(self, samples: torch.FloatTensor) -> torch.float64:
        # Gather indices of node values from one-hot arrays
        node_1_vals = torch.argmax(samples[:, self.node_1, :], dim=-1)
        node_2_vals = torch.argmax(samples[:, self.node_2, :], dim=-1)

        # Marginal
        normalizer = torch.logsumexp(self.P_1, dim=0)
        marginal = self.P_1[node_1_vals] - normalizer

        # Conditional
        cond_normalizer = torch.logsumexp(self.P_2_1, dim=0)
        conditional = self.P_2_1[node_2_vals, node_1_vals] - cond_normalizer[node_1_vals]

        return marginal + conditional

    '''
    Set all model parameters to maximize sample likelihood.
    Used for pretraining on source distribution before transfer training.
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    '''
    def set_maximum_likelihood(self, samples: torch.FloatTensor) -> None:
        with torch.no_grad():
            sample_occurrences = samples[:, self.node_1, :].reshape(-1, 1, self.N) * samples[:, self.node_2, :].reshape(-1, self.N, 1)
            occurrences = torch.sum(sample_occurrences, dim=0)
            self.P_1.data = torch.log(torch.sum(occurrences, dim=0) + EPS) - torch.log(torch.sum(occurrences) + EPS)
            self.P_2_1.data = torch.log(occurrences + EPS) - torch.log(torch.sum(occurrences, dim=0, keepdim=True) + EPS)


'''
Model for approximating two-variable relationship as completely
independent, with factorized probabilities.
Given samples drawn from a distribution, outputs their log likelihood.
'''
class IndependentModel(nn.Module, HypothesisModel):
    def __init__(self, N, M, node_1_ind, node_2_ind):
        nn.Module.__init__(self)

'''
Model for approximating two-variable relationship as an undirected
association, with a single table of shared likelihoods.
Given samples drawn from a distribution, outputs their log likelihood.
'''
class AssociatedModel(nn.Module, HypothesisModel):
    def __init__(self, N, M, node_1_ind, node_2_ind):
        nn.Module.__init__(self)




# --- OVERALL STRUCTURAL MODEL ---

'''
Interface for a structural model which can store and meta-train the causal
relationship between two variables from a multivariate functional causal model
'''
class BinarySubsetStructuralModel(nn.Module):
    def __init__(self):
        super(BinarySubsetStructuralModel, self).__init__()

    '''
    Forward pass computes the online log likelihood of the given samples,
    given the current parameters of each hypothesis model, weighed by softmax
    of gamma, which is the current likelihood of each hypothesis being true
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    Returns:
        (torch.FloatTensor):            Overall log likelihood of the given samples under
                                        this model's current parameters.
    '''
    def forward(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    '''
    Set all hypothesis model parameters to maximize likelihood of given samples.
    Used as pretraining before hypothesis models train on transfer distribution.
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    '''
    def set_maximum_likelihood(self, samples: torch.FloatTensor) -> None:
        raise NotImplementedError

    def reset_structure_parameters(self) -> None:
        raise NotImplementedError

    def structure_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError

    def hypothesis_models(self) -> List[HypothesisModel]:
        raise NotImplementedError

    def structure_likelihoods(self) -> List[float]:
        raise NotImplementedError

    def hypothesis_names(self) -> List[str]:
        raise NotImplementedError

    def hypothesis_count(self) -> int:
        raise NotImplementedError

    # Non-abstract functions
    def hypothesis_parameters(self) -> List[nn.Parameter]:
        params = []
        for h_model in self.hypothesis_models():
            params += list(h_model.parameters())
        return params
    

class CauseOnlyBinaryStructureModel(BinarySubsetStructuralModel):
    def __init__(self, N: int, M: int, a_ind: int, b_ind: int) -> None:
        super(CauseOnlyBinaryStructureModel, self).__init__()

        self.N = N

        # Hypothesis models
        self.HYPOTHESIS_COUNT = 2
        self.model_A_B = CauseModel(N, M, a_ind, b_ind)
        self.model_B_A = CauseModel(N, M, b_ind, a_ind)
        #self.model_independent = IndependentModel(N, M, a_ind, b_ind)
        #self.model_association = AssociatedModel(N, M, a_ind, b_ind)

        # Structural parameters to measure likelihood of each hypothesis
        self.gamma = nn.Parameter(torch.zeros(self.HYPOTHESIS_COUNT))

    
    def forward(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        log_model_weights = self.gamma - torch.logsumexp(self.gamma, dim=0)

        # Weighted log likelihood for each hypothesis
        model_A_B_logL = log_model_weights[0] + torch.sum(self.model_A_B(samples))
        model_B_A_logL = log_model_weights[1] + torch.sum(self.model_B_A(samples))
        
        # Remove max contribution
        shared_logL = torch.max(model_A_B_logL, model_B_A_logL)
        remaining_model_A_B_logL = model_A_B_logL - shared_logL
        remaining_model_B_A_logL = model_B_A_logL - shared_logL

        return shared_logL + torch.log(
            torch.exp(remaining_model_A_B_logL) +
            torch.exp(remaining_model_B_A_logL)
        )


    def set_maximum_likelihood(self, samples: torch.FloatTensor) -> None:
        self.model_A_B.set_maximum_likelihood(samples)
        self.model_B_A.set_maximum_likelihood(samples)

    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.gamma]

    def reset_structure_parameters(self) -> None:
        self.gamma.data = torch.zeros(self.HYPOTHESIS_COUNT)

    def hypothesis_models(self) -> List[HypothesisModel]:
        return [
            self.model_A_B,
            self.model_B_A
        ]

    def structure_likelihoods(self) -> torch.FloatTensor:
        return torch.exp(self.gamma - torch.logsumexp(self.gamma, dim=0)).detach()

    def hypothesis_names(self) -> List[str]:
        return ["A -> B", "B -> A"]

    def hypothesis_count(self) -> int:
        return 2