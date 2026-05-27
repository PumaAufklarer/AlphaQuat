from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # Data
    seq_length: int = 60
    stride: int = 10
    n_features: int = 14
    n_bins: int = 101
    price_range: float = 0.20

    # Model
    d_model: int = 128
    nhead: int = 4
    n_layers: int = 4
    dim_feed: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # Date ranges
    train_start: str = "20200101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240630"
