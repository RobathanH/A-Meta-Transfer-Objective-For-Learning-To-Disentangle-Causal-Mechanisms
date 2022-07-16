# Transfer Objective for Multivariate Causal Structure Learning

Robathan Harries, Rohan Virani, Arvind Sridhar

## Stanford CS-330 Project

This repository was forked from the original repository used in Bengio et al, but we ended up reimplementing most of the code from scratch. All of our additions, including the completed project report, can be found in the cs330 directory.

This course project implemented and tested potential multivariate extensions of binary methods detailed in 'A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms' (Bengio et al, 2019). Using concepts from meta-learning, the original methods pre-trained candidate models which assumed different causal structures (A -> B vs B -> A), and compared their adaptation speed when fine-tuned on transfer distributions (in which root/independent nodes in the ground-truth functional causal graph changed distributions, but edge/dependent nodes were unchanged). Models which embody the correct causal structure should adapt more quickly, since fewer parameters need to be adjusted.

Our course project investigated two potential extensions from the binary case to the unbounded multivariate case:
* Pairwise Binary Transfer Objective:
    * We simply apply the same binary algorithm to each pair of nodes in the graph
    * This method tends to work correctly for directly-connected node pairs (A -> B), but hallucinates causal directions for mediated node pairs (A -> Z -> B) and behaves unpredictably for confounded pairs (A <- Z -> B), colliding pairs (A -> Z <- B) and unrelated pairs (A ... B).
* Causal Parent Multivariate Transfer Objective:
    * Suggested in Appendix F of Bengio et al.
    * Expands the binary transfer objective regret function to the case of arbitrarily large graphs, then factors the problem of determining full causal graphs into independent sub-problems of determining the causal parents for each individual node, which can be computed simultaneously.
    * By factoring the problem in this way, the likelihoods of hypotheses A -> B and B -> A (which are meta-learned over multiple transfer episodes) no longer compete, and the algorithm is reduced to only learning associations rather than causal directions.
    * In experiments, this method is good at distinguishing directly-connected node pairs from indirectly-connected and unconnected pairs, but entirely fails to determine causal direction.

### Future Work:

With more time, we would have liked to investigate the performance of an algorithm which combines both approaches, isolating direct connections using the Causal Parent method, and determining pairwise causal directions using Pairwise Binary method.