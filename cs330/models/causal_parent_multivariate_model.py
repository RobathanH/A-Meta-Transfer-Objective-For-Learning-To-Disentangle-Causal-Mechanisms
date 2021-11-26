from typing import Iterable, List
from itertools import chain
from cs330.models.augmented_binary_models import CauseOnlyBinaryStructureModel
import numpy as np

import torch
import torch.nn as nn

from ..data_generators.multivariate_categorical import NodeFunction

EPS = 1e-9 # to avoid divide-by-zero error in log

'''
Method 1: Originally suggested in Appendix F of Bengio et al (2019)

Since this algorithm works independently to determine the parent of each
node in the graph, this model implementation only considers one node at
a time.
'''
class CausalParentMultivariateModel:
    '''
    Args:
        N (int):                        Number of values for each categorical variable
        M (int):                        Total number of variables in the causal graph
        target_node_ind (int):          Graph index for the node assumed to be the cause
    '''
    def __init__(self, N: int, M: int, target_node_ind: int) -> None:
        self.N = N
        self.M = M
        self.target_node_ind = target_node_ind

        # Inner Parameters
        # 2 layer neural network to parametrize the functional model of the node
        # given its parents.
        # (Uses the exact same parameters as the underlying function in the data generator)
        self.node_function = nn.Sequential(
            nn.Linear(N * M, 4 * max(N, M)),
            nn.ReLU(),
            nn.Linear(4 * max(N, M), N)
        )
        self.pretrained_node_function_state = None

        # Outer Parameters
        # parametrize causal structure with array of pre-sigmoid likelihoods that
        # each node is a parent of the target node
        self.parent_weights = nn.Parameter(torch.zeros(self.M))

    '''
    Sample a single hypothesis (which nodes are parents of target node) using the parent weights
    as Bernoulli probabilities. Shape = (M,). Element i is 1 if node i is parent of target node.
    '''
    def sample_hypothesis(self):
        parent_likelihoods = torch.sigmoid(self.parent_weights)
        parent_likelihoods[self.target_node_ind] = 0 # Ensure that the target node will not be listed as its own parent

        return torch.bernoulli(parent_likelihoods).detach()

    '''
    Saves the current hypothesis model state to be reset at the start of each transfer episode 
    '''
    def save_pretrained_node_function(self):
        self.pretrained_node_function_state = self.node_function.state_dict()

    '''
    Sets the current hypothesis model to match the saved pretrained state
    '''
    def load_pretrained_node_function(self):
        if self.pretrained_node_function_state is None:
            raise ValueError("Cannot load pretrained node function, since it was never saved")
        self.node_function.load_state_dict(self.pretrained_node_function_state)

    '''
    Computes the log likelihood of the given samples under the current hypothesis,
    and the current node function parameters. Returns the computed log likelihood,
    allowing it to be backpropogated to train the node function, and accumulated for 
    use in the regret function.
    Args:
        hypothesis (torch.FloatTensor): Current sampled hypothesis, same format as returned
                                        from sample_hypothesis().
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    Returns:
        (torch.FloatTensor):            Singleton log likelihood of the given samples under
                                        the current hypothesis and node function parameters.
    '''
    def hypothesis_log_likelihood(self, hypothesis: torch.FloatTensor, samples: torch.FloatTensor) -> torch.FloatTensor:
        # Ensure target node is masked out just to be safe, though it should be by default
        if hypothesis[self.target_node_ind] != 0:
            hypothesis[self.target_node_ind] = 0
        
        flattened_samples = samples.reshape(-1, self.N * self.M)
        masked_flattened_samples = flattened_samples * torch.repeat_interleave(hypothesis, self.N)
        node_output_probabilities = self.node_function(masked_flattened_samples)
        logL = samples[:, self.target_node_ind, :] * (node_output_probabilities - torch.logsumexp(node_output_probabilities, dim=1, keepdim=True))
        return torch.sum(logL)

    '''
    Manually computes structural regret derivatives with respect to structural parameters (parent_weights).
    Sets the gradients for all structural parameters, so they can be updated through optimizer step.
    This effectively does regret.backward().
    Uses torch.no_grad, so tensor arguments need not be attached to computation graph.
    Manual gradient computation is required since we are sampling from a bernoulli distribution,
    and taking the derivative with respect to those bernoulli probabilities.
    Args:
        hypothesis_list (torch.FloatTensor):                    Set of sampled hypotheses. Shape = (K, M), contains only
                                                                1's and 0's. K is the number of hypotheses sampled, and must
                                                                be the same between both arguments.
        hypothesis_online_log_likelihoods (torch.FloatTensor):  Set of online log likelihoods accumulated while training each
                                                                hypothesis on samples from a transfer episode. Shape = (K,).                                
    '''
    def compute_structure_gradients(self, hypothesis_list: torch.FloatTensor, hypothesis_online_log_likelihoods: torch.FloatTensor) -> None:
        with torch.no_grad():
            grad = torch.matmul(
                torch.sigmoid(self.parent_weights).reshape(-1, 1) - hypothesis_list.transpose(0, 1),    # shape = (M, K)
                torch.softmax(hypothesis_online_log_likelihoods, dim=0)                                 # shape = (K)
            )
            grad[self.target_node_ind] = 0
            self.parent_weights.grad = grad

    '''
    Reset structure parameters
    '''
    def reset_structure_parameters(self) -> None:
        self.parent_weights.data = torch.zeros(self.M)

    '''
    Return list of structure parameters (for outer-loop training)
    '''
    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.parent_weights]

    '''
    Return list of hypothesis parameters (for inner-loop training)
    '''
    def hypothesis_parameters(self) -> List[nn.Parameter]:
        return self.node_function.parameters()

    # --- Logging and info methods ---

    '''
    Return current structure likelihoods (chance that each node is parent of target node)
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        probs = torch.sigmoid(self.parent_weights).detach()
        probs[self.target_node_ind] = 0
        return probs


'''
Like above, uses the original Bengio et al suggested method, but applies it to learn all node parents at once.
'''

class FullGraphCausalParentMultivariateModel:
    '''
    Args:
        N (int):                        Number of values for each categorical variable
        M (int):                        Total number of variables in the causal graph
    '''
    def __init__(self, N: int, M: int) -> None:
        self.N = N
        self.M = M

        # Inner Parameters
        # 2 layer neural network to parametrize the functional model of the node
        # given its parents.
        # (Uses the exact same parameters as the underlying function in the data generator)
        self.node_functions = nn.ModuleList()
        for _ in range(M):
            self.node_functions.append(
                nn.Sequential(
                    nn.Linear(N * M, 4 * max(N, M)),
                    nn.ReLU(),
                    nn.Linear(4 * max(N, M), N)
                )
            )
        self.pretrained_state = None

        # Outer Parameters
        # parametrize causal structure with array of pre-sigmoid likelihoods that
        # each node is a parent of the target node
        # Element (i, j) gives pre-sigmoid likelihood that node i is directly caused by node j
        self.parent_weights = nn.Parameter(torch.zeros(self.M, self.M))

    '''
    Sample a single hypothesis (which nodes are parents of target node) using the parent weights
    as Bernoulli probabilities. Shape = (M,). Element i is 1 if node i is parent of target node.
    '''
    def sample_hypothesis(self):
        parent_likelihoods = torch.sigmoid(self.parent_weights)
        # Ensure nodes cannot be causes of themselves
        parent_likelihoods *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)

        return torch.bernoulli(parent_likelihoods).detach()

    '''
    Saves the current hypothesis model state to be reset at the start of each transfer episode 
    '''
    def save_pretrained(self):
        self.pretrained_state = self.node_functions.state_dict()

    '''
    Sets the current hypothesis model to match the saved pretrained state
    '''
    def load_pretrained(self):
        if self.pretrained_state is None:
            raise ValueError("Cannot load pretrained node function, since it was never saved")
        self.node_functions.load_state_dict(self.pretrained_state)

    '''
    Computes the log likelihood of the given samples under the current hypothesis,
    and the current node function parameters. Returns the computed log likelihood,
    allowing it to be backpropogated to train the node function, and accumulated for 
    use in the regret function.
    Args:
        hypothesis (torch.FloatTensor): Current sampled hypothesis, same format as returned
                                        from sample_hypothesis() (MxM shape filled with 1s, 0s).
        samples (torch.FloatTensor):    Samples from some causal distribution.
                                        Each sample in the batch contains M one-hot
                                        arrays of size N, each corresponding to a
                                        categorical value
                                        Shape = (batch_size, M, N)
    Returns:
        (torch.FloatTensor):            Log likelihoods of the given samples under
                                        the current hypothesis and node function parameters,
                                        one element for each predicted node.
                                        Shape = (M,)
    '''
    def hypothesis_log_likelihood(self, hypothesis: torch.FloatTensor, samples: torch.FloatTensor) -> torch.FloatTensor:
        # Ensure target node is masked out just to be safe, though it should be by default
        hypothesis = hypothesis * (torch.ones(self.M, self.M) - torch.eye(self.M, self.M))
        
        # Format input for each node function, masking to include only the hypothesized parents for each node
        batch_size = samples.shape[0]
        flattened_samples = samples.reshape(batch_size, self.N * self.M)
        node_output_probabilities = torch.zeros(batch_size, self.M, self.N)
        for target_node in range(self.M):
            masked_flattened_samples = flattened_samples * torch.repeat_interleave(hypothesis[target_node, :], self.N)
            node_output_probabilities[:, target_node, :] = self.node_functions[target_node](masked_flattened_samples)
        logL = samples * (node_output_probabilities - torch.logsumexp(node_output_probabilities, dim=2, keepdim=True))
        return torch.sum(logL, dim=(0, 2))


    '''
    Manually computes structural regret derivatives with respect to structural parameters (parent_weights).
    Sets the gradients for all structural parameters, so they can be updated through optimizer step.
    This effectively does regret.backward().
    Uses torch.no_grad, so tensor arguments need not be attached to computation graph.
    Manual gradient computation is required since we are sampling from a bernoulli distribution,
    and taking the derivative with respect to those bernoulli probabilities.
    Args:
        hypothesis_list (torch.FloatTensor):                    Set of sampled hypotheses. Shape = (K, M, M), contains only
                                                                1's and 0's. K is the number of hypotheses sampled, and must
                                                                be the same between both arguments.
        hypothesis_online_log_likelihoods (torch.FloatTensor):  Set of online log likelihoods accumulated while training each
                                                                hypothesis on samples from a transfer episode. Shape = (K, M).
                                                                For each sampled hypothesis, stored online log likelihoods for
                                                                each target node.                               
    '''
    def compute_structure_gradients(self, hypothesis_list: torch.FloatTensor, hypothesis_online_log_likelihoods: torch.FloatTensor) -> None:
        with torch.no_grad():
            # Reshape to treat each target node individually
            hypothesis_list = hypothesis_list.permute(1, 2, 0)                                                          # Shape = (M, M, K) (target_node, parent_node, hypothesis_sample_number)
            hypothesis_online_log_likelihoods = hypothesis_online_log_likelihoods.permute(1, 0).reshape(self.M, 1, -1)  # Shape = (M, 1, K) (target_node, null, hypothesis_sample_number)
            
            # We only need relative online likelihoods (no log) of each sample hypothesis (for each target node)
            hypothesis_relative_online_L = torch.softmax(hypothesis_online_log_likelihoods, dim=1)                      # Shae = (M, M, K)

            # Base gradients are determined by difference between bernoulli probability (sigmoid) and hypothesis value, before being summed over
            # all hypotheses, weighted by the relative online likelihoods of each hypothesis
            base_grad = torch.sigmoid(self.parent_weights).reshape(self.M, self.M, 1) - hypothesis_list                 # Shape = (M, M, K)

            grad = torch.sum(base_grad * hypothesis_relative_online_L, dim=2)                                           # Shape = (M, M)

            # Force self-parent weights to have 0 gradient
            grad *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)

            self.parent_weights.grad = grad

    '''
    Reset structure parameters
    '''
    def reset_structure_parameters(self) -> None:
        self.parent_weights.data = torch.zeros(self.M, self.M)

    '''
    Return list of structure parameters (for outer-loop training)
    '''
    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.parent_weights]

    '''
    Return list of hypothesis parameters (for inner-loop training)
    '''
    def hypothesis_parameters(self) -> List[nn.Parameter]:
        return self.node_functions.parameters()

    # --- Logging and info methods ---

    '''
    Return current structure likelihoods (chance that each node is parent of target node)
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        probs = torch.sigmoid(self.parent_weights).detach()
        probs *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)
        return probs