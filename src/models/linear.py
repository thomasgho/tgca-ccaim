import torch
from torch import nn


class RegressionModel(nn.Module):
    """
    A generic linear regression model.
    
    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    """

    def __init__(self, in_dim, out_dim):
        super(RegressionModel, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
    
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        Reinitialize learnable parameters via normal distribution.
        
        """
        nn.init.normal_(self.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.bias)    
    
    
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
        outputs = torch.matmul(inputs, self.weight)
        outputs += self.bias
        
        return outputs
    
    
    @staticmethod
    def loss(pred, target):
        """
        Regression loss, chosen to be mean squared error.
        
        Parameters
        ----------
        pred : torch.Tensor
            The predicted output.
        target : torch.Tensor
            The target output.

        Returns
        -------
        torch.Tensor
            
        """
        criterion = nn.MSELoss(reduction='mean')
        
        return criterion(pred, target)