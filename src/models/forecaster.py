"""
Deep learning models for volatility forecasting (LSTM + Self-Attention).
"""

import logging
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class IVSequenceDataset(Dataset):
    """Dataset for IV forecasting sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature sequences (n_samples, seq_len, n_features)
            y: Targets (n_samples,)
            w: Sample weights (n_samples,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.w = torch.from_numpy(w).float()
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.w[idx]


class LSTMSelfAttentionIVForecaster(nn.Module):
    """
    LSTM with self-attention mechanism for IV forecasting.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Self-attention mechanism
        self.attn_W = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # LSTM encoding
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        
        # Self-attention
        u = torch.tanh(self.attn_W(out))
        scores = self.attn_v(u).squeeze(-1)  # (batch, seq_len)
        alpha = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        
        # Prediction
        out = self.fc(context).squeeze(-1)  # (batch,)
        
        return out, alpha


class IVForecasterTrainer:
    """
    Trainer for IV forecasting models.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to run on (default: cuda if available, else cpu)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int = 100, lr: float = 1e-3, patience: int = 10) -> Tuple[list, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            
        Returns:
            Tuple of (train_losses, val_losses)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            cooldown=0,
            min_lr=1e-6,
        )
        
        best_val_loss = np.inf
        best_state = None
        epochs_no_improve = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, n_epochs + 1):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            n_train_batches = 0
            
            for Xb, yb, wb in train_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                
                optimizer.zero_grad()
                preds, _ = self.model(Xb)
                mse = (preds - yb)**2
                loss = (mse * wb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                n_train_batches += 1
            
            epoch_train_loss /= max(1, n_train_batches)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for Xb, yb, wb in val_loader:
                    Xb = Xb.to(self.device)
                    yb = yb.to(self.device)
                    wb = wb.to(self.device)
                    
                    preds, _ = self.model(Xb)
                    mse = (preds - yb)**2
                    val_loss_batch = (mse * wb).mean()
                    
                    epoch_val_loss += val_loss_batch.item()
                    n_val_batches += 1
            
            epoch_val_loss /= max(1, n_val_batches)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            logger.info(f'Epoch {epoch:03d} | train_loss={epoch_train_loss:.6f} | '
                       f'val_loss={epoch_val_loss:.6f}')
            
            scheduler.step(epoch_val_loss)
            
            # Early stopping
            if epoch_val_loss < best_val_loss - 1e-6:
                best_val_loss = epoch_val_loss
                best_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (y_true, y_pred, weights)
        """
        self.model.eval()
        y_true = []
        y_pred = []
        w_all = []
        
        with torch.no_grad():
            for Xb, yb, wb in test_loader:
                Xb = Xb.to(self.device)
                
                preds, _ = self.model(Xb)
                preds = preds.cpu().numpy()
                
                y_true.append(yb.numpy())
                y_pred.append(preds)
                w_all.append(wb.numpy())
        
        if len(y_true) > 0:
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            w_all = np.concatenate(w_all)
        else:
            y_true = np.array([])
            y_pred = np.array([])
            w_all = np.array([])
        
        return y_true, y_pred, w_all



