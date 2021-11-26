from typing import Dict, List
from enum import Enum
import numpy as np

import torch
import torch.nn as nn

# Constants
EPS = 1e-9 # to avoid divide-by-zero error in log

'''
Enum representing each possible binary relationship between two nodes.
Termed as hypotheses, since structural model contains multiple models,
each assuming and parametrizing a particular one of these relationships.
'''
class Hypothesis(Enum):
    FORWARD_CAUSE = 1
    BACKWARD_CAUSE = 2
    INDEPENDENT = 3

    def name(self):
        if self is Hypothesis.FORWARD_CAUSE:
            return "A -> B"
        if self is Hypothesis.BACKWARD_CAUSE:
            return "B -> A"
        if self is Hypothesis.INDEPENDENT:
            return "A || B"

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
class BinarySubsetStructuralModel:
    def __init__(self):
        super(BinarySubsetStructuralModel, self).__init__()

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
    def pretrain_hypotheses(self, samples: torch.FloatTensor) -> None:
        raise NotImplementedError

    '''
    Computes the log likelihood of the given samples under each hypothesis,
    given the current parameters of each hypothesis model. Returns the
    current log likelihood for each current hypothesis model, allowing them to
    be backpropogated to train the hypothesis models, and accumulated for 
    use in the regret function.
    Args:
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    Returns:
        (torch.FloatTensor):            Log likelihood of the given samples under
                                        each hypothesis model's current parameters.
    '''
    def hypothesis_log_likelihoods(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    '''
    Use the accumulated online log likelihood for each hypothesis model to return
    the regret value, which weighs online losses by the structure parameters for
    each hypothesis.
    Args:
        hypothesis_online_log_likelihoods (torch.FloatTensor):  Sum of log likelihoods over training
                                                                for each hypothesis, with list elements
                                                                corresponding to hypothesis order returned in
                                                                hypotheses() and hypothesis_log_likelihoods().
    Returns:
        (torch.FloatTensor):                                    Regret loss function for meta-training step.
    '''
    def structure_regret(self, hypothesis_online_log_likelihoods: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    '''
    Reset all structure parameters so all possible hypotheses have equal weights.
    '''
    def reset_structure_parameters(self) -> None:
        raise NotImplementedError

    # --- Logging and info methods

    '''
    Returns all parameters for weighing between hypotheses
    '''
    def structure_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError

    '''
    Returns all parameters for independent hypothesis models
    '''
    def hypothesis_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError

    '''
    Returns the number of hypotheses considered by this structure model.
    Corresponds to the length of many returned lists and first dim of many
    returned tensors.
    '''
    def hypothesis_count(self) -> int:
        raise NotImplementedError

    '''
    Returns list of Hypothesis enum corresponding to each hypothesis model
    in order.
    '''
    def hypotheses(self) -> List[Hypothesis]:
        raise NotImplementedError

    '''
    Returns detached tensor listing current structure parameter likelihoods
    for each hypothesis.
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        raise NotImplementedError
    

class CauseOnlyBinaryStructureModel(BinarySubsetStructuralModel):
    def __init__(self, N: int, M: int, a_ind: int, b_ind: int) -> None:
        super(CauseOnlyBinaryStructureModel, self).__init__()

        self.N = N

        # Hypothesis list (corresponds to model lists and structure)
        self.hypothesis_list = [
            Hypothesis.FORWARD_CAUSE,
            Hypothesis.BACKWARD_CAUSE
        ]

        # Hypothesis models
        self.hypothesis_models = [
            CauseModel(N, M, a_ind, b_ind),
            CauseModel(N, M, b_ind, a_ind)
        ]

        # Structural parameters to measure likelihood of each hypothesis
        # These weights are passed through softmax to get structure likelihoods
        self.hypothesis_weights = nn.Parameter(torch.zeros(self.hypothesis_count()))

    def pretrain_hypotheses(self, samples: torch.FloatTensor) -> None:
        for model in self.hypothesis_models:
            model.set_maximum_likelihood(samples)

    def hypothesis_log_likelihoods(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        # Compute current log likelihood for each hypothesis, and
        # return dict of hypothesis losses for backpropagation
        result = torch.zeros(self.hypothesis_count())
        for i in range(self.hypothesis_count()):
            result[i] = torch.sum(self.hypothesis_models[i](samples))
        return result

    def structure_regret(self, hypothesis_online_log_likelihoods: List[float]) -> torch.FloatTensor:
        log_softmax_weights = self.hypothesis_weights - torch.logsumexp(self.hypothesis_weights, dim=0)

        # Save structure-weighted log likelihood for each hypothesis
        weighted_hypothesis_online_logL = log_softmax_weights + hypothesis_online_log_likelihoods
        
        # Regret = negative log of weighted sum of online likelihoods
        return -torch.logsumexp(weighted_hypothesis_online_logL, dim=0)

    def reset_structure_parameters(self) -> None:
        self.hypothesis_weights.data = torch.zeros(self.hypothesis_count())

    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.hypothesis_weights]

    def hypothesis_parameters(self) -> List[nn.Parameter]:
        params = []
        for model in self.hypothesis_models:
            params += list(model.parameters())
        return params

    def hypotheses(self) -> List[Hypothesis]:
        return self.hypothesis_list
    
    def hypothesis_count(self) -> int:
        return len(self.hypothesis_list)

    def structure_likelihoods(self) -> torch.FloatTensor:
        return torch.exp(self.hypothesis_weights - torch.logsumexp(self.hypothesis_weights, dim=0)).detach()