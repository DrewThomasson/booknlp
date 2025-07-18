"""
Complete training system for the BERTSpeakerID model.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import the existing BERTSpeakerID model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from booknlp.english.speaker_attribution import BERTSpeakerID

from .config import TrainingConfig
from .data_utils import SpeakerAttributionDataProcessor

logger = logging.getLogger(__name__)

class SpeakerAttributionDataset(Dataset):
    """PyTorch Dataset for speaker attribution data."""
    
    def __init__(self, data: List[Dict[str, Any]], processor: SpeakerAttributionDataProcessor):
        self.data = data
        self.processor = processor
        self.samples = []
        
        # Prepare all training samples
        for sample in data:
            batch_x, batch_m = processor.prepare_training_batch([sample])
            for x, m in zip(batch_x, batch_m):
                self.samples.append((x, m))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class SpeakerAttributionTrainer:
    """Main trainer class for speaker attribution model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.processor = SpeakerAttributionDataProcessor(config)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.early_stopping_counter = 0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.logs_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load and split data into train/validation sets."""
        logger.info(f"Loading data from {data_path}")
        
        # Load dataset
        data = self.processor.load_dataset(data_path)
        
        # Validate all samples
        valid_data = []
        for i, sample in enumerate(data):
            if self.processor.validate_sample(sample):
                valid_data.append(sample)
            else:
                logger.warning(f"Skipping invalid sample at index {i}")
        
        logger.info(f"Loaded {len(valid_data)} valid samples out of {len(data)} total")
        
        # Get dataset statistics
        stats = self.processor.get_dataset_statistics(valid_data)
        logger.info(f"Dataset statistics: {stats}")
        
        # Split data
        train_data, val_data = train_test_split(
            valid_data,
            test_size=1 - self.config.train_test_split,
            random_state=self.config.random_seed,
            shuffle=True
        )
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def initialize_model(self, base_model_path: Optional[str] = None):
        """Initialize the BERTSpeakerID model."""
        try:
            if base_model_path and os.path.exists(base_model_path):
                logger.info(f"Loading model from {base_model_path}")
                self.model = torch.load(base_model_path, map_location=self.device)
            else:
                logger.info(f"Initializing new model with base model: {self.config.base_model}")
                self.model = BERTSpeakerID(base_model=self.config.base_model)
            
            self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            logger.info("Model and optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def create_data_loaders(self, train_data: List[Dict], val_data: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders."""
        train_dataset = SpeakerAttributionDataset(train_data, self.processor)
        val_dataset = SpeakerAttributionDataset(val_data, self.processor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch):
        """Custom collate function for batching data."""
        batch_x = [item[0] for item in batch]
        batch_m = [item[1] for item in batch]
        return batch_x, batch_m
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (batch_x, batch_m) in enumerate(train_loader):
            try:
                # Get model batches
                x_batches, m_batches, y_batches, o_batches = self.model.get_batches(
                    batch_x, batch_m, batch_size=len(batch_x)
                )
                
                batch_loss = 0.0
                batch_correct = 0
                batch_total = 0
                
                for x_batch, m_batch, y_batch in zip(x_batches, m_batches, y_batches):
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = self.model(x_batch, m_batch)
                    
                    # Calculate loss
                    targets = y_batch["y"]
                    loss = self.criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Track metrics
                    batch_loss += loss.item()
                    predicted_labels = torch.argmax(predictions, dim=-1)
                    batch_correct += (predicted_labels == targets).sum().item()
                    batch_total += targets.numel()
                    
                    self.global_step += 1
                    
                total_loss += batch_loss
                total_correct += batch_correct
                total_samples += batch_total
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {batch_loss:.4f}, Acc: {batch_correct/batch_total:.4f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_m in val_loader:
                try:
                    # Get model batches
                    x_batches, m_batches, y_batches, o_batches = self.model.get_batches(
                        batch_x, batch_m, batch_size=len(batch_x)
                    )
                    
                    for x_batch, m_batch, y_batch in zip(x_batches, m_batches, y_batches):
                        # Forward pass
                        predictions = self.model(x_batch, m_batch)
                        
                        # Calculate loss
                        targets = y_batch["y"]
                        loss = self.criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))
                        total_loss += loss.item()
                        
                        # Collect predictions and targets
                        predicted_labels = torch.argmax(predictions, dim=-1)
                        all_predictions.extend(predicted_labels.cpu().numpy().flatten())
                        all_targets.extend(targets.cpu().numpy().flatten())
                        
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        if all_predictions and all_targets:
            accuracy = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0
            )
        else:
            accuracy = precision = recall = f1 = 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.model_output_dir, 'best_model.pt')
            torch.save(self.model, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self, data_path: str):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Set seed for reproducibility
        self._set_seed(self.config.random_seed)
        
        # Load and prepare data
        train_data, val_data = self.load_data(data_path)
        train_loader, val_loader = self.create_data_loaders(train_data, val_data)
        
        # Initialize model
        self.initialize_model()
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Check for best model
            current_score = val_metrics['f1']
            is_best = current_score > self.best_score
            if is_best:
                self.best_score = current_score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info(f"Training completed. Best F1 score: {self.best_score:.4f}")
        
        # Save final model
        final_path = os.path.join(self.config.model_output_dir, 'final_model.pt')
        torch.save(self.model, final_path)
        logger.info(f"Saved final model to {final_path}")
        
        return {"best_f1": self.best_score, "final_epoch": self.current_epoch}