"""
Configuration settings for speaker attribution training.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration class for speaker attribution training."""
    
    # Model settings
    base_model: str = "bert-base-uncased"
    max_sequence_length: int = 512
    max_candidates: int = 10
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # Data settings
    train_test_split: float = 0.8
    random_seed: int = 42
    
    # Paths
    data_dir: str = "./data"
    model_output_dir: str = "./models"
    logs_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # Logging
    log_level: str = "INFO"
    save_every_n_epochs: int = 1
    evaluate_every_n_steps: int = 500
    
    # BookNLP integration
    booknlp_model_path: Optional[str] = None
    spacy_model: str = "en_core_web_sm"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.model_output_dir, self.logs_dir, self.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

@dataclass
class DatasetConfig:
    """Configuration for dataset creation GUI."""
    
    # File paths
    default_text_file: str = ""
    default_output_file: str = "dataset.json"
    
    # GUI settings
    window_width: int = 1200
    window_height: int = 800
    font_family: str = "Arial"
    font_size: int = 11
    
    # Text highlighting colors
    quote_color: str = "#ffeb3b"  # Yellow
    entity_color: str = "#4caf50"  # Green
    selected_color: str = "#2196f3"  # Blue
    
    # BookNLP pipeline settings
    pipeline_components: List[str] = field(default_factory=lambda: ["entity", "quote"])
    auto_detect_entities: bool = True
    auto_detect_quotes: bool = True
    
    # Entity types to include
    entity_types: List[str] = field(default_factory=lambda: ["PERSON", "PER"])