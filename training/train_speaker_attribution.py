"""
Main training script for speaker attribution model.
Command-line interface for training the BERTSpeakerID model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfig
from training.data_utils import SpeakerAttributionDataProcessor

# Import trainer only when needed (to avoid torch import issues in validation-only mode)
SpeakerAttributionTrainer = None

def setup_logging(log_level: str, log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from file."""
    if not os.path.exists(config_path):
        logging.info(f"Config file {config_path} not found, using defaults")
        return TrainingConfig()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return TrainingConfig.from_dict(config_dict)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        logging.info("Using default configuration")
        return TrainingConfig()

def save_config(config: TrainingConfig, config_path: str):
    """Save training configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving config to {config_path}: {e}")

def validate_data_file(data_path: str) -> bool:
    """Validate that the data file exists and is properly formatted."""
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return False
    
    try:
        processor = SpeakerAttributionDataProcessor()
        data = processor.load_dataset(data_path)
        
        if not data:
            logging.error("Dataset is empty")
            return False
        
        # Validate a few samples
        valid_count = 0
        for i, sample in enumerate(data[:10]):  # Check first 10 samples
            if processor.validate_sample(sample):
                valid_count += 1
            else:
                logging.warning(f"Sample {i} is invalid")
        
        if valid_count == 0:
            logging.error("No valid samples found in dataset")
            return False
        
        logging.info(f"Dataset validation passed: {len(data)} samples, {valid_count}/10 samples checked are valid")
        return True
        
    except Exception as e:
        logging.error(f"Error validating data file: {e}")
        return False

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train speaker attribution model using BERTSpeakerID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to training data JSON file"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.json",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="./logs",
        help="Directory to save training logs"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="bert-base-uncased",
        help="Base BERT model to use"
    )
    
    # Device and performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training"
    )
    
    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    # Resume training
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    # Validation
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset without training"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the configuration to file"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.data_dir = os.path.dirname(args.data_path)
    config.model_output_dir = args.output_dir
    config.logs_dir = args.logs_dir
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.base_model = args.base_model
    config.device = args.device
    config.mixed_precision = args.mixed_precision
    config.early_stopping_patience = args.early_stopping_patience
    config.log_level = args.log_level
    
    # Setup logging
    setup_logging(config.log_level, config.logs_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting speaker attribution training")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.config)
    
    # Validate data file
    if not validate_data_file(args.data_path):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)
    
    # If validation only, exit here
    if args.validate_only:
        logger.info("Data validation completed successfully.")
        sys.exit(0)
    
    try:
        # Import trainer only when needed
        global SpeakerAttributionTrainer
        if SpeakerAttributionTrainer is None:
            from training.speaker_attribution_trainer import SpeakerAttributionTrainer
        
        # Initialize trainer
        trainer = SpeakerAttributionTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            if os.path.exists(args.resume_from):
                logger.info(f"Resuming training from {args.resume_from}")
                trainer.initialize_model(args.resume_from)
            else:
                logger.error(f"Checkpoint file not found: {args.resume_from}")
                sys.exit(1)
        
        # Start training
        results = trainer.train(args.data_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best F1 score: {results['best_f1']:.4f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save final results
        results_path = os.path.join(config.model_output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Training results saved to {results_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()