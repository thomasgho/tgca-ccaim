import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

class EarlyStopper:
    """
    Early stopping via test loss monitoring.
    """
    def __init__(self, patience=1, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_test_loss = np.inf
        self.test_metric = 0.

    def early_stop(self, test_loss, test_metric):
        
        if test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.test_metric = test_metric
            self.counter = 0
        
        elif test_loss > (self.min_test_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False


class Trainer:
    """
    Train routine helper.
    """
    
    def __init__(self, model, optimizer, metric, patience, accumulation_steps, 
                 log_dir, cuda=True, checkpoint=False, mixed_precision=False, is_vae=False):
        super().__init__()
        
        # CUDA tensors
        if cuda:
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            self.device = torch.device("cpu")
        
        # training instances
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.metric = metric.to(self.device)
        
        # early stopping
        self.early_stopper = EarlyStopper(patience=patience, min_delta=0.)
        
        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        
        # tensorboard
        self.log_dir = log_dir
        
        # checkpointing
        self.checkpoint = checkpoint
        
        # automatic mixed precision training
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        # if model is VAE (requiers modified input/loss setup)
        self.is_vae = is_vae

    
    def fit(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            writer = SummaryWriter(log_dir=self.log_dir)
            
            train_loss, train_metric = self._train(train_loader)
            test_loss, test_metric = self._test(test_loader)
            
            # checkpoint model
            if self.checkpoint:
                torch.save(self.model.state_dict(), 
                           self.log_dir + "/checkpoint_epoch_{}".format(epoch))
            
            # print updates
            print("Epoch {}: \n \
            -> Train Loss = {} \n \
            -> Test Loss = {} \n \
            -> Train Metric = {} \n \
            -> \033[1mTest Metric = {}\033[0m".format(
                epoch,
                train_loss,
                test_loss,
                train_metric,
                test_metric))
            
            # update tensorboard
            writer.add_scalars("Loss", {"Train": train_loss,
                                        "Test": test_loss}, epoch)
            writer.add_scalars("Metric", {"Train": train_metric,
                                        "Test": test_metric}, epoch)
            
            # check early stop
            if self.early_stopper.early_stop(test_loss, test_metric):             
                print(
                    "Applying early stopping... \n \
                Minimum Test Loss acheived on Epoch {}: \n \
                -> \033[1mFinal Test Metric = {}\033[0m".format(
                        epoch - self.early_stopper.patience,
                        self.early_stopper.test_metric,
                    )
                )
                break
        
        # write pending to disk
        writer.flush()
        writer.close()

        
    def fit_metric(self):
        return self.early_stopper.test_metric
    
    
    def _train(self, loader):
        self.model.train()
        
        batch_losses = []
        batch_metrics = []
        
        for batch, batch_data in enumerate(loader):
            
            # send tensors to device
            feats = batch_data["feats"].to(self.device)
            outcome = batch_data["outcome"].to(self.device)

            # forward propagation (mixed precision if needed)
            with torch.autocast(device_type=str(self.device), 
                                dtype=torch.float16, 
                                enabled=self.mixed_precision):
                
                if self.is_vae:
                    # separating gene feats and treatment context
                    pred_outcome, recon_inputs, mu, logvar = self.model(
                        feats[:, :-1], feats[:, -1])
                    pred_outcome = pred_outcome.squeeze()
                    loss = self.model.loss(
                        feats[:, :-1], recon_inputs, mu, logvar, 
                        outcome, pred_outcome, beta=1.)
                else:
                    pred_outcome = self.model(feats).squeeze()
                    loss = self.model.loss(pred_outcome, outcome)
            
            # backward pass on scaled gradients
            self.scaler.scale(loss).backward()
            
            # evaluate metric
            metric = self.metric(pred_outcome, outcome)
            
            # update weights after accumulation_steps iterations
            # effective batch size is increased if needed (low memory devices)
            if (batch + 1) % self.accumulation_steps == 0 or (batch + 1) == len(loader.datapipe):
                # update and reset (unscaled) gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=False)
            
                # append to list
                batch_losses.append(loss.item())
                batch_metrics.append(metric.item())
        
        # compute average over batches
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_metric = sum(batch_metrics) / len(batch_metrics)
        
        return avg_loss, avg_metric

    
    def _test(self, loader):
        self.model.eval()

        with torch.no_grad():
            
            batch_losses = []
            batch_metrics = [] 
            
            for batch_data in loader:
                
                # send tensors to device
                feats = batch_data["feats"].to(self.device)
                outcome = batch_data["outcome"].to(self.device)
                
                # forward propagation
                if self.is_vae:
                    # separating gene feats and treatment context
                    pred_outcome, recon_inputs, mu, logvar = self.model(
                        feats[:, :-1], feats[:, -1])
                    pred_outcome = pred_outcome.squeeze()
                    loss = self.model.loss(
                        feats[:, :-1], recon_inputs, mu, logvar, 
                        outcome, pred_outcome, beta=1.)
                else:
                    pred_outcome = self.model(feats).squeeze()
                    loss = self.model.loss(pred_outcome, outcome)
                
                # evaluate metric
                metric = self.metric(pred_outcome, outcome)
                
                # append to list
                batch_losses.append(loss.item())
                batch_metrics.append(metric.item())
                
            # compute average over batches
            avg_loss = sum(batch_losses) / len(batch_losses)
            avg_metric = sum(batch_metrics) / len(batch_metrics)
            
        return avg_loss, avg_metric