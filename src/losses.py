# Adapted from https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import combinations


class SinkhornSolver(nn.Module):
    """
    Optimal Transport solver under entropic regularisation.
    Based on the code of Gabriel Peyré.
    """
    def __init__(self, epsilon, iterations=100, ground_metric=lambda x: torch.pow(x, 2)):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.ground_metric = ground_metric

    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)
        
        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y
        
        a = a.squeeze()
        b = b.squeeze()
                
        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Stopping criterion
        threshold = 1e-1
        
        # Cost matrix
        C = self._compute_cost(x, y)
        
        # Sinkhorn iterations
        for i in range(self.iterations): 
            u0, v0 = u, v
                        
            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
            u = self.epsilon * u_ + u
                        
            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
            v = self.epsilon * v_ + v
            
            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)
                        
            if mean_diff.item() < threshold:
                break
   
        print("Finished computing transport plan in {} iterations".format(i))
    
        # Transport plan pi = diag(a)*K*diag(b)
        K = self._log_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)
        
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        return cost, pi

    def _compute_cost(self, x, y):
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.ground_metric(x_ - y_), dim=-1)
        return C

    def _log_boltzmann_kernel(self, u, v, C=None):
        C = self._compute_cost(u, v) if C is None else C
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel
    

class NTBXentLoss:
    def __init__(self, temperature):
        self.temperature=temperature

    def __call__(self, x, codes):
        return self.nt_bxent_loss(x, codes)

    @staticmethod
    def find_positive_pairs(codes):
        if codes is None:
            return torch.tensor([])
        code_positions = {}
        positive_pairs = []
        # Loop through the list and store all positions of each code
        for i, code in enumerate(codes):
            if code in code_positions:
                code_positions[code].append(i)
            else:
                code_positions[code] = [i]
        # For each code that has multiple occurrences, create all possible pairs
        for positions in code_positions.values():
            if len(positions) > 1:
                # Generate all combinations of pairs from positions
                positive_pairs.extend(list(combinations(positions, 2)))
        return torch.tensor(positive_pairs)

    def nt_bxent_loss(self, x, codes):
        assert len(x.size()) == 2
        device = x.device
        dtype = x.dtype

        # Add indexes of the principal diagonal elements to pos_indices
        pos_indices = self.find_positive_pairs(codes)
        if pos_indices.any():
            pos_indices = torch.cat([
                pos_indices,
                torch.arange(x.size(0)).reshape(x.size(0), 1).expand(-1, 2),
            ], dim=0)
        else:
            pos_indices = torch.arange(x.size(0)).reshape(x.size(0), 1).expand(-1, 2)
            print("pos_indices", pos_indices)
        
        # Ground truth labels
        target = torch.zeros(x.size(0), x.size(0))
        target[pos_indices[:,0], pos_indices[:,1]] = 1.0
        target = target.to(dtype=dtype, device=device)

        # Cosine similarity
        xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
        # Set logit of diagonal element to "inf" signifying complete
        # correlation. sigmoid(inf) = 1.0 so this will work out nicely
        # when computing the Binary Cross Entropy Loss.
        xcs[torch.eye(x.size(0)).bool()] = float("inf")

        # Standard binary cross entropy loss. We use binary_cross_entropy() here and not
        # binary_cross_entropy_with_logits() because of https://github.com/pytorch/pytorch/issues/102894
        # The method *_with_logits() uses the log-sum-exp-trick, which causes inf and -inf values
        # to result in a NaN result.

        loss = F.binary_cross_entropy((xcs / self.temperature).sigmoid(), target, reduction="none")
        
        target_pos = target.bool()
        target_neg = ~target_pos
        try:
            loss_pos = torch.zeros((x.size(0), x.size(0)),
                                dtype=dtype,
                                device=x.device).masked_scatter(target_pos, loss[target_pos])
            loss_neg = torch.zeros((x.size(0), x.size(0)),
                            dtype=dtype,
                            device=x.device).masked_scatter(target_neg, loss[target_neg])
        except RuntimeError:
            import pdb; pdb.set_trace()
            
        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)
        num_pos = target.sum(dim=1)
        num_neg = x.size(0) - num_pos

        return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
