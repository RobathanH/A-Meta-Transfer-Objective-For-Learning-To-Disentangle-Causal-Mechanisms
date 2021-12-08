from typing import List
import numpy as np
import itertools
from collections import deque

import torch
import torch.nn as nn

'''
Defines the format for parametrizing the full graph
'''
def graph_function_architecture(N, M):
    return nn.ModuleList([
        nn.Sequential(
            nn.Linear(N * M, 4 * max(N, M)),
            nn.ReLU(),
            nn.Linear(4 * max(N, M), N)
        )
        for _ in range(M)
    ])




# Computes all causal graph directed adjacency matrices (with no self-loops)
def all_causal_graphs(M):
    all_candidate_hypotheses = np.stack([np.array(unique_binary).reshape(M, M) for unique_binary in itertools.product([0, 1], repeat=M * M)])
    return all_candidate_hypotheses[np.trace(np.abs(all_candidate_hypotheses), axis1=1, axis2=2) == 0]

# Computes all acyclic causal graph directed adjacency matrices
def all_acyclic_causal_graphs(M):
    # Begin with all possible directed adjacency matrices
    all_candidate_hypotheses = all_causal_graphs(M)
    invalid_candidates = np.zeros(len(all_candidate_hypotheses))

    # Compute all paths up to length M, invalidating candidates which produce cycles
    path_tracker = np.broadcast_to(np.identity(M), all_candidate_hypotheses.shape)

    # Max possible cycle has M edges
    for _ in range(M):
        path_tracker = all_candidate_hypotheses @ path_tracker
        invalid_candidates += np.trace(np.abs(path_tracker), axis1=1, axis2=2)

    valid_hypotheses = all_candidate_hypotheses[invalid_candidates == 0, :, :]
    return valid_hypotheses

# Computes all unique one-node causal parent graphs, repeated for each node in the graph
def all_causal_parent_graphs(M):
    all_hypotheses = np.zeros((2**(M - 1), M, M))
    for i, unique_binary in enumerate(itertools.product([0, 1], repeat=M - 1)):
        unshifted_single_node_hypothesis = np.array([0] + list(unique_binary))
        for target_node in range(M):
            all_hypotheses[i, target_node] = np.roll(unshifted_single_node_hypothesis, target_node)
    return all_hypotheses



'''
Originally suggested in Appendix F of Bengio et al (2019)

Treat each target node as an independent function, and learn the expected
causal parents of each target node independently.
'''
class CausalParentMultivariateModel:
    '''
    Args:
        N (int):                        Number of values for each categorical variable
        M (int):                        Total number of variables in the causal graph
        hypothesis_sample_count (int):  Number of hypotheses to sample for each outer 
                                        loop iteration.
    '''
    def __init__(self, N: int, M: int, hypothesis_sample_count: int) -> None:
        self.N = N
        self.M = M
        self.hypothesis_sample_count = hypothesis_sample_count

        # Inner Parameters
        # 2 layer neural network to parametrize the functional model of the node
        # given its parents.
        # (Uses the exact same parameters as the underlying function in the data generator)
        self.node_functions = graph_function_architecture(N, M)
        self.pretrained_state = None

        # Outer Parameters
        # parametrize causal structure with array of pre-sigmoid likelihoods that
        # each node is a parent of the target node
        # Element (i, j) gives pre-sigmoid likelihood that node i is directly caused by node j
        self.structure_weights = nn.Parameter(torch.zeros(self.M, self.M))

    '''
    Return the hypothesis (or hypotheses) which should be pretrained. In the Causal Parent model,
    one node function is pretrained on the most permissive possible hypothesis (all parents possible)
    and used for transfer training on all sampled hypotheses.
    Returns stacked hypothesis tensors, shape = (..., M, M). 
    Element i,j is 1 if node i is directly caused by node j
    '''
    def pretrain_hypotheses(self) -> torch.Tensor:
        return (torch.ones(self.M, self.M) - torch.eye(self.M, self.M)).reshape(1, self.M, self.M).detach()

    '''
    Saves the current hypothesis model after pretraining on the current distribution (before intervention).
    Args:
        pretrain_hypothesis (torch.Tensor): The hypothesis from the list from pretrain_hypotheses() which the
                                            current model was trained under.
    '''
    def save_pretrained(self, pretrain_hypothesis: torch.Tensor) -> None:
        self.pretrained_state = self.node_functions.state_dict()

    '''
    Sample all hypotheses (which nodes are parents of target node) to be considered in
    one outer loop iteration of this structural model. 
    Shape = (hypothesis_sample_count, M, M). Element i,j is 1 if node i is directly caused by node j.
    '''
    def sample_hypotheses(self) -> torch.Tensor:
        parent_likelihoods = torch.sigmoid(self.structure_weights)
        # Ensure nodes cannot be causes of themselves
        parent_likelihoods *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)

        hypotheses = torch.zeros(self.hypothesis_sample_count, self.M, self.M).detach()
        for hypothesis_iter in range(self.hypothesis_sample_count):
            hypotheses[hypothesis_iter] = torch.bernoulli(parent_likelihoods).detach()

        return hypotheses

    '''
    Sets the current hypothesis model to match the saved pretrained state, in preparation for transfer training
    on the given hypothesis_to_test.
    Args:
        hypothesis_to_test (torch.Tensor):  The hypothesis model which the loaded pretrained state should correspond
                                            to. Should be one of the hypotheses returned from sample_hypotheses().
                                            In the base Causal Parent multivariate model, there is only one
                                            pretrained state for all models, but this is not always the case for 
                                            other multivariate models.
    '''
    def load_pretrained(self, hypothesis_to_test: torch.Tensor) -> None:
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
    Manually computes structural regret derivatives with respect to structural parameters (structure_weights).
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
            hypothesis_relative_online_L = torch.softmax(hypothesis_online_log_likelihoods, dim=2)                      # Shape = (M, 1, K)

            # Base gradients are determined by difference between bernoulli probability (sigmoid) and hypothesis value, before being summed over
            # all hypotheses, weighted by the relative online likelihoods of each hypothesis
            base_grad = torch.sigmoid(self.structure_weights).reshape(self.M, self.M, 1) - hypothesis_list                 # Shape = (M, M, K)

            grad = torch.sum(base_grad * hypothesis_relative_online_L, dim=2)                                           # Shape = (M, M)

            # Force self-parent weights to have 0 gradient
            grad *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)            

            self.structure_weights.grad = grad

    '''
    Reset structure parameters
    '''
    def reset_structure_parameters(self) -> None:
        self.structure_weights.data = torch.zeros(self.M, self.M)

    '''
    Return list of structure parameters (for outer-loop training)
    '''
    def structure_parameters(self) -> List[nn.Parameter]:
        return [self.structure_weights]

    '''
    Return list of hypothesis parameters (for inner-loop training)
    '''
    def hypothesis_parameters(self) -> List[nn.Parameter]:
        return self.node_functions.parameters()

    # --- Logging and info methods ---

    '''
    Return the current predicted most-likely full-graph hypothesis
    '''
    def predicted_graph(self) -> torch.FloatTensor:
        return torch.round(torch.sigmoid(self.structure_weights)).detach()

    '''
    Return current structure likelihoods (chance that each node is parent of target node)
    '''
    def structure_likelihoods(self) -> torch.FloatTensor:
        probs = torch.sigmoid(self.structure_weights).detach()
        probs *= torch.ones(self.M, self.M) - torch.eye(self.M, self.M)
        return probs

    '''
    Returns names identifying each value in structure_likelihoods().
    Args:
        index:  Tuple of integers indexing a particular value in structure_likelihoods().
    Returns:
        (str):  A short string identifying the meaning of that structure parameter index.
    '''
    def structure_param_name(self, *index) -> str:
        target, parent = index
        return f"Node {parent} -> Node {target}"


'''
Version of the Causal Parent Multivariate Model which samples all hypotheses, skipping
the biased monte-carlo hypothesis sampling step at the cost of intractable scaling.
'''
class AllHypotheses_CausalParentMultivariateModel(CausalParentMultivariateModel):
    '''
    Same format as CausalParentMultivariateModel parameters, but hypothesis_sample_count
    is not used for anything.
    '''
    def __init__(self, N: int, M: int, hypothesis_sample_count: int):
        super(AllHypotheses_CausalParentMultivariateModel, self).__init__(N, M, hypothesis_sample_count)

        # Constants
        # Repeat each hypothesis multiple times for stability
        self.HYPOTHESIS_REPEATS = 3

        # Compute list of all possible hypotheses.
        # Since each target node's structure is learned independently, exhaustive list must
        # only cover each possible set of parents, thus 2^(M - 1) entries
        self.all_hypotheses = torch.from_numpy(all_causal_parent_graphs(self.M)).detach()

    '''
    Return all possible hypotheses, not just a repeatedly sampled set of hypotheses
    '''
    def sample_hypotheses(self):
        return self.all_hypotheses.repeat(self.HYPOTHESIS_REPEATS, 1, 1)

    '''
    Compute structure gradients based on explicit expectation over hypotheses.
    Without the sampling approximation, gradient can be computed automatically.
    '''
    def compute_structure_gradients(self, hypothesis_list: torch.FloatTensor, hypothesis_online_log_likelihoods: torch.FloatTensor) -> None:
        # Compute the current likelihood of each hypothesis, treating each target node hypothesis individually
        hypothesis_structure_log_likelihoods = torch.sum(hypothesis_list * torch.log(torch.sigmoid(self.structure_weights)).reshape(1, self.M, self.M) + (1 - hypothesis_list) * torch.log(1 - torch.sigmoid(self.structure_weights)).reshape(1, self.M, self.M), dim=2)

        # Regret for each target node is the negative log of the weighted sum (expected value) of online likelihood across all hypotheses
        # Sum regret over each independent target node to compute loss
        regret = -torch.sum(torch.logsumexp(hypothesis_structure_log_likelihoods + hypothesis_online_log_likelihoods, dim=0))

        regret.backward()