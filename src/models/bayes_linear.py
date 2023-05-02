import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class BayesRegressionModel(nn.Module):
    """
    Bayesian linear layer, with a Gaussian prior on weights.
    
    Parameters
    ----------    
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    mu : float
        Mean of prior distribution.
    sigma : float
        Std of prior distribution.
    bias : bool
        Whether to use an additive bias.

    """
    def __init__(self, in_dim, out_dim, mu, sigma, bias=True):
        super(BayesRegressionModel, self).__init__()
        
        self.mu = mu
        self.sigma = sigma
        self.log_sigma = math.log(sigma)
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.register_buffer('weight_eps', None)
        
        self.bias = bias
            
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_dim))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_dim))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()

        
    def reset_parameters(self):
        """
        Reinitialize learnable parameters via method outlined in:
        https://arxiv.org/abs/1810.01279
        
        """
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.log_sigma)
        
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.log_sigma)
            

    def forward(self, inputs):
        """
        Forward computation
        
        Parameters
        ----------
        inputs : torch.Tensor
            The input feature.

        Returns
        -------
        torch.Tensor
            The output tensor.
            
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
        else :
            bias = None
            
        return F.linear(inputs, weight, bias)
    
    
    def loss(self, pred_outputs, outputs):
        """
        Calculates KL divergence on layer weights.
        Additional mean squared error term is used for regression.
        
        """
        mse = nn.MSELoss()(pred_outputs, outputs)
        
        kld = (self.log_sigma - self.weight_log_sigma + \
        (torch.exp(self.weight_log_sigma) ** 2 + \
         (self.mu - self.weight_mu) ** 2) / (2 * math.exp(self.log_sigma) ** 2) - 0.5).sum()
        n = len(self.weight_mu.view(-1))
        
        if self.bias:
            kld += (self.log_sigma - self.bias_log_sigma + \
            (torch.exp(self.bias_log_sigma) ** 2 + \
             (self.mu - self.bias_mu) ** 2) / (2 * math.exp(self.log_sigma) ** 2) - 0.5).sum()
            n += len(self.bias_mu.view(-1))

        kld /= n
        
        return mse + 0.1 * kld
        