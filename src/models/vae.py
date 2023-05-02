import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    
    Parameters
    ----------
    dim : int
        Residual dimension size.
    batch_norm : bool
        Whether to use batch normalization.
    
    """
    def __init__(self, dim, batch_norm=True):
        super().__init__()
        
        self.linear_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(2)])
        
        self.relu = nn.ReLU()
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(dim, eps=1e-3) for _ in range(2)])
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """
        Reinitialize second layer weights.
        
        """
        nn.init.normal_(self.linear_layers[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.linear_layers[-1].bias)
        
        
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
            The output feature.
            
        """        
        
        temps = inputs
        
        if self.batch_norm:
            temps = self.batch_norm_layers[0](temps)
        
        temps = self.relu(temps)
        temps = self.linear_layers[0](temps)
        
        if self.batch_norm:
            temps = self.batch_norm_layers[1](temps)
        
        temps = self.relu(temps)
        temps = self.linear_layers[1](temps)
        
        return inputs + temps


class Encoder(nn.Module):
    """
    Encoder network for VAE with residual connections. 
    Outputs mean and log-variance for Gaussian reparametrization.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    z_dim : int
        Latent dimension size.
    hidden_dim : int
        Hidden dimension size.
    num_blocks : int   
        Number of residual blocks.
    batch_norm : bool
        Whether to use batch normalization.
    """

    def __init__(self, in_dim, z_dim, hidden_dim, num_blocks=2, 
                 batch_norm=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.init_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim=hidden_dim, batch_norm=batch_norm)
            for _ in range(num_blocks)])
        self.final_layer_loc = nn.Linear(hidden_dim, z_dim)
        self.final_layer_scale = nn.Linear(hidden_dim, z_dim)        
        
        
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
            The output feature.
            
        """    
        temps = self.init_layer(inputs)
        
        for block in self.blocks:
            temps = block(temps)
        
        mu = self.final_layer_loc(temps)
        logvar = self.final_layer_scale(temps)
        
        return mu, logvar

    
class Decoder(nn.Module):
    """
    Decoder network for VAE with residual connections. 

    Parameters
    ----------
    out_dim : int
        Output feature size.
    z_dim : int
        Latent dimension size.
    hidden_dim : int
        Hidden dimension size.
    num_blocks : int   
        Number of residual blocks.
    batch_norm : bool
        Whether to use batch normalization.
    """

    def __init__(self, out_dim, z_dim, hidden_dim, num_blocks=2, 
                 batch_norm=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.init_layer = nn.Linear(z_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim=hidden_dim, batch_norm=batch_norm)
            for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_dim, out_dim)


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
            The output feature.
            
        """        
        temps = self.init_layer(inputs)
        
        for block in self.blocks:
            temps = block(temps)
        
        outputs = self.final_layer(temps)
        
        return outputs
    
    
class BetaVAEModel(nn.Module):
    """
    Variational Autoencoder with Gaussian prior from
    https://arxiv.org/abs/1312.6114 and 
    https://www.deepmind.com/publications/beta-vae-learning-
    basic-visual-concepts-with-a-constrained-variational-framework
    
    An additional simple two-layer network is used to 
    predict the outcomes from the latent variables.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    z_dim : int
        Latent dimension size.
    hidden_dim : int
        Hidden dimension size.
    num_blocks : int   
        Number of residual blocks.
    batch_norm : bool
        Whether to use batch normalization.
    """

    def __init__(self, in_dim, out_dim, z_dim, hidden_dim, context_dim, 
                 num_blocks=2, batch_norm=True):
        super().__init__()
        
        self.encoder = Encoder(
            in_dim = in_dim, 
            z_dim = z_dim, 
            hidden_dim = hidden_dim, 
            num_blocks = num_blocks, 
            batch_norm = batch_norm,
        )
        
        self.decoder = Decoder(
            out_dim = in_dim, 
            z_dim = z_dim, 
            hidden_dim = hidden_dim, 
            num_blocks = num_blocks, 
            batch_norm = batch_norm,
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(z_dim + context_dim, z_dim // 2),
            nn.BatchNorm1d(z_dim // 2, eps=1e-3),
            nn.ReLU(),
            nn.Linear(z_dim // 2, out_dim),
        )
    
    
    def reparameterise(self, mu, logvar):
        """
        Reparameterization trick for Gaussian prior.
        
        Parameters
        ----------
        mu : torch.Tensor
            The predicted mean.
        logvar : torch.Tensor
            The predicted log of variance.
            
        Returns
        -------
        torch.Tensor
            The latent sample.
            
        """     
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    
    def forward(self, inputs, context):
        """
        Forward computation
        
        Parameters
        ----------
        inputs : torch.Tensor
            The input feature.
        context : torch.Tensor
            The context feature (treatment).
            
        Returns
        -------
        torch.Tensor
            The output feature.
            
        """        
        mu, logvar = self.encoder(inputs)
        z = self.reparameterise(mu, logvar)
        
        context = context.view(-1, 1)
        z_context = torch.cat([z, context], dim=1)
        outputs = self.predictor(z_context)
        
        recon_inputs = self.decoder(z)
        return outputs, recon_inputs, mu, logvar

    
    def latent_space(self, inputs):
        """
        Helper to draw samples from latent space.
        
        Parameters
        ----------
        inputs : torch.Tensor
            The input feature.

        Returns
        -------
        torch.Tensor
            The output feature.
            
        """        
        mu, logvar = self.encoder(inputs)
        z = self.reparameterise(mu, logvar)
        
        return z 

    
    @staticmethod
    def loss(inputs, recon_inputs, mu, logvar, 
             outputs, pred_outputs, beta):
        """
        ELBO loss with additional MSE term for 
        outcome predictior network.
        
        Parameters
        ----------
        inputs : torch.Tensor
            The input feature.
        recon_inputs : torch.Tensor
            The reconstructed input feature.
        mu : torch.Tensor
            The predicted latent mean.
        logvar : torch.Tensor
            The predicted log of variance.
        outputs : torch.Tensor
            The target output.   
        pred_outputs : torch.Tensor
            The predicted output (from predictor network).   
            
        Returns
        -------
        torch.Tensor
            
        """    
        mse_pred = nn.MSELoss(reduction='mean')(pred_outputs, outputs)
        mse_recon = nn.MSELoss(reduction='mean')(recon_inputs, inputs)
        kld = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return mse_pred + mse_recon + beta * kld
    