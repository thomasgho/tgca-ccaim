import torch
from torch import nn


class MLPModel(nn.Module):
    """
    A generic 2-layer MLP model.
    
    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(MLPModel, self).__init__()
        
        self.layer_1 = nn.Linear(in_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        
        """
        nn.init.xavier_normal_(self.layer_1.weight)
        nn.init.xavier_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)
        nn.init.zeros_(self.layer_2.bias)
        
    
    def forward(self, inputs):
        """
        Forward computation
        
        Parameters
        ----------
        x : torch.Tensor
            The input feature.

        Returns
        -------
        torch.Tensor
            The output tensor.
            
        """
        outputs = self.layer_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer_2(outputs)
        
        return outputs
    
    
    @staticmethod
    def loss(pred, target):
        """
        Mean squared error loss.
        
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