# Speaker Attribution Training System

A complete training system for the speaker attribution BERT model in BookNLP. This system provides tools for creating datasets, training models, and managing the complete training pipeline for speaker attribution tasks.

## Overview

The speaker attribution training system consists of several key components:

1. **Training Infrastructure** - Complete training loop with proper optimization, checkpointing, and evaluation
2. **Dataset Creation GUI** - Interactive tool for creating training datasets with entity detection integration
3. **Data Processing Utilities** - Tools for handling the specific JSON format required for training
4. **Configuration Management** - Centralized configuration for all training parameters
5. **Command-line Interface** - Easy-to-use CLI for training and validation

## Features

- ✅ Uses existing `BERTSpeakerID` model architecture
- ✅ Supports specified JSON data format with quote tokens and entity attribution
- ✅ GUI for interactive dataset creation with entity auto-detection
- ✅ Production-ready training loop with early stopping and checkpointing
- ✅ Cross-platform compatibility using tkinter
- ✅ Comprehensive error handling and logging
- ✅ Integration with existing BookNLP entity detection pipelines

## Installation

### Requirements

The training system requires the base BookNLP dependencies plus some additional packages:

```bash
# Install base BookNLP
pip install -e .

# Install additional training dependencies
pip install -r training/requirements.txt
```

### Additional Dependencies

- `scikit-learn>=1.3.0` - For metrics and data splitting
- `matplotlib>=3.4.0` - For visualization
- `tensorboard>=2.13.0` - For training monitoring
- `tkinter` - For GUI (usually included with Python)

## Quick Start

### 1. Run the Demo

```bash
cd /path/to/booknlp
python training/demo.py
```

This will create example data and demonstrate the system functionality.

### 2. Create a Dataset

Launch the dataset creator GUI:

```bash
python training/dataset_creator_gui.py
```

Or create a dataset programmatically using the data utilities.

### 3. Train a Model

```bash
python training/train_speaker_attribution.py path/to/your/dataset.json
```

## Data Format

The system uses a specific JSON format for training data:

```json
[
    {
        "text": ["John", "walked", "into", "the", "room", ".", "[QUOTE]", "Hello", "everyone", "!", "[/QUOTE]", "he", "said", "with", "a", "smile", "."],
        "quotes": [[6, 10]],
        "entities": [
            [0, 1, "PERSON", "John"],
            [11, 12, "PERSON", "he"]
        ],
        "attributions": {
            "0": 0
        }
    }
]
```

### Format Specification

- **`text`**: Tokenized text with special quote tokens `[QUOTE]` and `[/QUOTE]`
- **`quotes`**: List of `[start, end]` indices for quote spans (token-level)
- **`entities`**: List of `[start, end, type, text]` for detected entities (token-level)
- **`attributions`**: Dictionary mapping quote indices to entity indices

## Usage

### Dataset Creation GUI

The GUI provides an interactive interface for creating training datasets:

1. **Load Text**: Import text files for processing
2. **Entity Detection**: Automatically detect person entities using BookNLP
3. **Quote Detection**: Automatically detect quoted speech
4. **Attribution**: Link quotes to entities through the interface
5. **Dataset Management**: Save and load datasets in the required format

```bash
python training/dataset_creator_gui.py
```

### Training

#### Basic Training

```bash
python training/train_speaker_attribution.py dataset.json
```

#### Advanced Training Options

```bash
python training/train_speaker_attribution.py dataset.json \
    --learning-rate 1e-5 \
    --batch-size 8 \
    --num-epochs 10 \
    --output-dir ./models \
    --early-stopping-patience 3 \
    --mixed-precision
```

#### Resume Training

```bash
python training/train_speaker_attribution.py dataset.json \
    --resume-from ./checkpoints/checkpoint_epoch_5.pt
```

#### Validate Dataset Only

```bash
python training/train_speaker_attribution.py dataset.json --validate-only
```

### Configuration

Training parameters can be configured through:

1. **Command-line arguments** (highest priority)
2. **Configuration file** (`training_config.json`)
3. **Default values** (lowest priority)

#### Save Configuration

```bash
python training/train_speaker_attribution.py dataset.json --save-config
```

This creates a `training_config.json` file with current settings.

## Components

### 1. `config.py`
Configuration classes for training and dataset creation:
- `TrainingConfig`: Training hyperparameters and settings
- `DatasetConfig`: GUI and dataset creation settings

### 2. `data_utils.py`
Data processing utilities:
- `SpeakerAttributionDataProcessor`: Main data processing class
- Dataset validation and statistics
- Format conversion utilities
- Integration with BookNLP pipelines

### 3. `speaker_attribution_trainer.py`
Main training infrastructure:
- `SpeakerAttributionTrainer`: Complete training system
- `SpeakerAttributionDataset`: PyTorch dataset wrapper
- Training loop with early stopping
- Model checkpointing and evaluation

### 4. `dataset_creator_gui.py`
Interactive GUI application:
- `DatasetCreatorGUI`: Main GUI application
- Text loading and entity detection
- Interactive attribution creation
- Dataset management and export

### 5. `train_speaker_attribution.py`
Command-line interface:
- Argument parsing and validation
- Configuration management
- Training orchestration
- Error handling and logging

## Model Integration

The training system integrates with the existing `BERTSpeakerID` model:

```python
from booknlp.english.speaker_attribution import BERTSpeakerID

# The trainer automatically uses this model
model = BERTSpeakerID(base_model="bert-base-uncased")
```

### Supported Base Models

- `bert-base-uncased`
- `bert-large-uncased` 
- Custom BERT models following the naming convention

## Output Files

### Training Output

- **Models**: `./models/best_model.pt`, `./models/final_model.pt`
- **Checkpoints**: `./checkpoints/checkpoint_epoch_N.pt`
- **Logs**: `./logs/training.log`
- **Results**: `./models/training_results.json`

### Dataset Output

- **Dataset**: Custom filename with `.json` extension
- **Configuration**: `training_config.json`

## Logging and Monitoring

The system provides comprehensive logging:

```bash
# View training logs
tail -f logs/training.log

# Use tensorboard (if available)
tensorboard --logdir logs/
```

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure BookNLP is properly installed and in Python path
2. **CUDA Issues**: Set `--device cpu` if GPU is not available
3. **Memory Issues**: Reduce `--batch-size`
4. **Data Validation Errors**: Check data format using `--validate-only`

### Getting Help

1. Run the demo: `python training/demo.py`
2. Check the logs in `./logs/training.log`
3. Use `--validate-only` to check data format
4. Review the example dataset created by the demo

## Examples

### Example 1: Complete Workflow

```bash
# 1. Create example data
python training/demo.py

# 2. Validate the data
python training/train_speaker_attribution.py example_data/example_dataset.json --validate-only

# 3. Train with custom settings
python training/train_speaker_attribution.py example_data/example_dataset.json \
    --learning-rate 2e-5 \
    --batch-size 4 \
    --num-epochs 5 \
    --output-dir ./my_models
```

### Example 2: Using the GUI

```bash
# Start the GUI
python training/dataset_creator_gui.py

# Then:
# 1. Load a text file
# 2. Run entity detection
# 3. Run quote detection  
# 4. Create attributions
# 5. Save dataset
# 6. Use saved dataset for training
```

## Development

### Running Tests

The system can be tested using the demo script:

```bash
python training/demo.py
```

### Adding New Features

1. Update configuration in `config.py`
2. Add data processing in `data_utils.py`
3. Extend training logic in `speaker_attribution_trainer.py`
4. Update CLI in `train_speaker_attribution.py`

## License

This training system is part of BookNLP and follows the same license terms.