"""PyTorch LSTM model with self-attention for latency prediction.

Architecture:
  - Input: (batch, seq_len=60, n_features≈45)
  - LSTM: 2 layers, hidden=128, dropout=0.2
  - Self-attention: single-head over LSTM outputs
  - Two output heads:
      latency_head  → (batch, 3) for t+10/30/60 s RTT
      congestion_head → (batch, 1) sigmoid for congestion prob
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from models.base_model import BaseLatencyModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTMModel unavailable")

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────────────────────
# PyTorch module definition
# ──────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _LSTMNetwork(nn.Module):
        """LSTM + self-attention network for latency and congestion."""

        def __init__(self, n_features: int, hidden_size: int = 128,
                     n_layers: int = 2, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                batch_first=True,
            )
            # Single-head self-attention
            self.attention_weight = nn.Linear(hidden_size, 1)

            # Output heads
            self.latency_head = nn.Linear(hidden_size, 3)
            self.congestion_head = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: (batch, seq_len, n_features).

            Returns:
                Tuple of (latency_preds, congestion_probs) with shapes
                (batch, 3) and (batch, 1).
            """
            lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

            # Attention weights
            attn_scores = self.attention_weight(lstm_out).squeeze(-1)  # (batch, seq)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
            context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)

            latency = self.latency_head(context)  # (batch, 3)
            congestion = self.congestion_head(context)  # (batch, 1)

            return latency, congestion


class LSTMModel(BaseLatencyModel):
    """LSTM model with attention for multi-horizon latency prediction.

    Normalises features with ``StandardScaler`` (fit on training data
    only) and stores the scaler alongside the model checkpoint.
    """

    SEQ_LEN = 60

    def __init__(self, n_features: int = 45, hidden_size: int = 128,
                 n_layers: int = 2, dropout: float = 0.2) -> None:
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._target_scaler: Optional[Any] = None
        self._feature_names: List[str] = []
        self._device = "cpu"

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"

    @property
    def name(self) -> str:
        return "lstm"

    def _create_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, step: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Create sliding-window sequences for the LSTM.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Targets (n_samples, 3+1): rtt horizons + congestion.
            step: Stride between windows.

        Returns:
            Tuple of (X_seq, y_latency, y_congestion) or
            (X_seq, None, None) if y is None.
        """
        sequences = []
        targets_lat = [] if y is not None else None
        targets_cong = [] if y is not None else None

        for i in range(0, len(X) - self.SEQ_LEN + 1, step):
            sequences.append(X[i : i + self.SEQ_LEN])
            if y is not None:
                idx = i + self.SEQ_LEN - 1
                targets_lat.append(y[idx, :3])
                targets_cong.append(y[idx, 3] if y.shape[1] > 3 else 0.0)

        X_seq = np.array(sequences, dtype=np.float32)
        y_lat = np.array(targets_lat, dtype=np.float32) if targets_lat else None
        y_cong = np.array(targets_cong, dtype=np.float32) if targets_cong else None

        return X_seq, y_lat, y_cong

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train the LSTM model.

        Args:
            X: Feature matrix.
            y: Target matrix [rtt_10s, rtt_30s, rtt_60s, congestion].
            **kwargs: Optional X_val, y_val for early stopping.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LSTMModel")

        # Fit scaler on training features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self.n_features = X.shape[1]
        self._model = _LSTMNetwork(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self._device)

        # Create sequences — use step>1 to reduce memory on large datasets.
        # step=5 gives 5× fewer sequences, cutting RAM from ~600 MB to ~120 MB.
        seq_step = kwargs.get("seq_step", max(1, len(X) // 20_000))
        X_seq, y_lat, y_cong = self._create_sequences(X_scaled, y, step=seq_step)
        del X_scaled  # free memory
        if X_seq.shape[0] == 0:
            logger.warning("Not enough data for LSTM sequences")
            return

        logger.info("LSTM sequences: {} (step={})", X_seq.shape[0], seq_step)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_lat, dtype=torch.float32),
            torch.tensor(y_cong, dtype=torch.float32),
        )
        del X_seq, y_lat, y_cong  # free numpy arrays, data lives in tensors now
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Validation set
        val_loader = None
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler.transform(X_val)
            Xv_seq, yv_lat, yv_cong = self._create_sequences(X_val_scaled, y_val, step=seq_step)
            del X_val_scaled
            if Xv_seq.shape[0] > 0:
                val_ds = TensorDataset(
                    torch.tensor(Xv_seq, dtype=torch.float32),
                    torch.tensor(yv_lat, dtype=torch.float32),
                    torch.tensor(yv_cong, dtype=torch.float32),
                )
                del Xv_seq, yv_lat, yv_cong
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        # Training loop
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        epochs = 50

        for epoch in range(epochs):
            self._model.train()
            train_loss = 0.0
            for X_batch, y_lat_batch, y_cong_batch in loader:
                X_batch = X_batch.to(self._device)
                y_lat_batch = y_lat_batch.to(self._device)
                y_cong_batch = y_cong_batch.to(self._device)

                optimizer.zero_grad()
                lat_pred, cong_pred = self._model(X_batch)

                loss_lat = mse_loss(lat_pred, y_lat_batch)
                loss_cong = bce_loss(cong_pred.squeeze(), y_cong_batch)
                loss = loss_lat + 0.3 * loss_cong

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validation
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_b, yl_b, yc_b in val_loader:
                        X_b = X_b.to(self._device)
                        yl_b = yl_b.to(self._device)
                        yc_b = yc_b.to(self._device)
                        lp, cp = self._model(X_b)
                        vl = mse_loss(lp, yl_b) + 0.3 * bce_loss(cp.squeeze(), yc_b)
                        val_loss += vl.item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping at epoch {}", epoch + 1)
                        break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch {}/{} — train_loss={:.4f}",
                    epoch + 1, epochs, train_loss / len(loader),
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RTT at three horizons.

        Args:
            X: Feature matrix (>= SEQ_LEN rows).

        Returns:
            Array of shape (n_pred, 3).
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted")

        self._model.eval()
        X_scaled = self._scaler.transform(X)
        X_seq, _, _ = self._create_sequences(X_scaled)

        if X_seq.shape[0] == 0:
            # Not enough data for a full sequence — use last SEQ_LEN rows
            if len(X_scaled) >= self.SEQ_LEN:
                X_seq = X_scaled[-self.SEQ_LEN:][np.newaxis, ...]
            else:
                return np.full((1, 3), 20.0)

        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
            lat_pred, _ = self._model(X_t)
            result = lat_pred.cpu().numpy()

        return np.clip(result, 0, None)

    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Predict congestion probability.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_pred,) with probabilities in [0, 1].
        """
        if self._model is None or self._scaler is None:
            return np.zeros(1)

        self._model.eval()
        X_scaled = self._scaler.transform(X)
        X_seq, _, _ = self._create_sequences(X_scaled)

        if X_seq.shape[0] == 0:
            if len(X_scaled) >= self.SEQ_LEN:
                X_seq = X_scaled[-self.SEQ_LEN:][np.newaxis, ...]
            else:
                return np.array([0.0])

        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
            _, cong_pred = self._model(X_t)
            return cong_pred.squeeze(-1).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights, scaler, and config.

        Args:
            path: File path for the checkpoint.
        """
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("No model to save")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "model_state": self._model.state_dict(),
            "scaler": self._scaler,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "feature_names": self._feature_names,
        }
        torch.save(checkpoint, path)
        logger.info("LSTMModel saved to {}", path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load a saved LSTMModel checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Restored LSTMModel instance.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to load LSTMModel")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_features=checkpoint["n_features"],
            hidden_size=checkpoint["hidden_size"],
            n_layers=checkpoint["n_layers"],
            dropout=checkpoint["dropout"],
        )
        model._scaler = checkpoint["scaler"]
        model._feature_names = checkpoint.get("feature_names", [])
        model._model = _LSTMNetwork(
            n_features=model.n_features,
            hidden_size=model.hidden_size,
            n_layers=model.n_layers,
            dropout=model.dropout,
        ).to(model._device)
        model._model.load_state_dict(checkpoint["model_state"])
        logger.info("LSTMModel loaded from {}", path)
        return model
