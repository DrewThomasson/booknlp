"""
Example and demo script for the speaker attribution training system.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.data_utils import SpeakerAttributionDataProcessor
from training.config import TrainingConfig

def create_example_dataset():
    """Create an example dataset for demonstration."""
    
    # Example data in the required format
    example_data = [
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
        },
        {
            "text": ["Mary", "responded", "quickly", ".", "[QUOTE]", "Nice", "to", "see", "you", "John", "!", "[/QUOTE]", "She", "smiled", "warmly", "."],
            "quotes": [[4, 11]],
            "entities": [
                [0, 1, "PERSON", "Mary"],
                [9, 10, "PERSON", "John"],
                [12, 13, "PERSON", "She"]
            ],
            "attributions": {
                "0": 0
            }
        },
        {
            "text": ["The", "old", "man", "shook", "his", "head", ".", "[QUOTE]", "I", "don't", "understand", "this", "at", "all", ".", "[/QUOTE]", "he", "muttered", "."],
            "quotes": [[7, 15]],
            "entities": [
                [1, 3, "PERSON", "old man"],
                [16, 17, "PERSON", "he"]
            ],
            "attributions": {
                "0": 0
            }
        }
    ]
    
    return example_data

def demo_data_processing():
    """Demonstrate data processing functionality."""
    print("=== Speaker Attribution Training System Demo ===\n")
    
    # Initialize processor
    processor = SpeakerAttributionDataProcessor()
    
    # Create example dataset
    print("1. Creating example dataset...")
    example_data = create_example_dataset()
    print(f"   Created {len(example_data)} example samples")
    
    # Validate samples
    print("\n2. Validating samples...")
    valid_count = 0
    for i, sample in enumerate(example_data):
        if processor.validate_sample(sample):
            valid_count += 1
            print(f"   Sample {i}: ✓ Valid")
        else:
            print(f"   Sample {i}: ✗ Invalid")
    
    print(f"   {valid_count}/{len(example_data)} samples are valid")
    
    # Get dataset statistics
    print("\n3. Dataset statistics:")
    stats = processor.get_dataset_statistics(example_data)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Prepare training batch
    print("\n4. Preparing training batch...")
    try:
        batch_x, batch_m = processor.prepare_training_batch(example_data[:1])
        print(f"   Prepared batch with {len(batch_x)} sequences")
        print(f"   First sequence length: {len(batch_x[0])}")
        print(f"   First sequence: {' '.join(batch_x[0][:10])}...")
    except Exception as e:
        print(f"   Error preparing batch: {e}")
    
    # Save example dataset
    print("\n5. Saving example dataset...")
    output_dir = Path("./example_data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "example_dataset.json"
    
    try:
        processor.save_dataset(example_data, output_file)
        print(f"   Saved to: {output_file}")
    except Exception as e:
        print(f"   Error saving dataset: {e}")
    
    # Load dataset back
    print("\n6. Loading dataset...")
    try:
        loaded_data = processor.load_dataset(output_file)
        print(f"   Loaded {len(loaded_data)} samples")
    except Exception as e:
        print(f"   Error loading dataset: {e}")

def demo_configuration():
    """Demonstrate configuration system."""
    print("\n=== Configuration Demo ===\n")
    
    # Create default config
    print("1. Default configuration:")
    config = TrainingConfig()
    config_dict = config.to_dict()
    
    for key, value in config_dict.items():
        print(f"   {key}: {value}")
    
    # Save config
    print("\n2. Saving configuration...")
    config_file = Path("./example_data/example_config.json")
    try:
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"   Saved to: {config_file}")
    except Exception as e:
        print(f"   Error saving config: {e}")
    
    # Load config
    print("\n3. Loading configuration...")
    try:
        loaded_config = TrainingConfig.from_dict(config_dict)
        print(f"   Loaded config with learning_rate: {loaded_config.learning_rate}")
    except Exception as e:
        print(f"   Error loading config: {e}")

def show_usage_examples():
    """Show usage examples for the training system."""
    print("\n=== Usage Examples ===\n")
    
    print("1. Training with default settings:")
    print("   python training/train_speaker_attribution.py example_data/example_dataset.json")
    
    print("\n2. Training with custom settings:")
    print("   python training/train_speaker_attribution.py example_data/example_dataset.json \\")
    print("          --learning-rate 1e-5 \\")
    print("          --batch-size 8 \\")
    print("          --num-epochs 5 \\")
    print("          --output-dir ./my_models")
    
    print("\n3. Validating dataset only:")
    print("   python training/train_speaker_attribution.py example_data/example_dataset.json --validate-only")
    
    print("\n4. Starting the dataset creator GUI:")
    print("   python training/dataset_creator_gui.py")
    
    print("\n5. Resuming training from checkpoint:")
    print("   python training/train_speaker_attribution.py example_data/example_dataset.json \\")
    print("          --resume-from ./checkpoints/checkpoint_epoch_5.pt")

def main():
    """Main demo function."""
    print("Speaker Attribution Training System")
    print("===================================")
    
    try:
        # Run data processing demo
        demo_data_processing()
        
        # Run configuration demo
        demo_configuration()
        
        # Show usage examples
        show_usage_examples()
        
        print("\n=== Demo completed successfully! ===")
        print("\nNext steps:")
        print("1. Create your own dataset using the GUI: python training/dataset_creator_gui.py")
        print("2. Train a model: python training/train_speaker_attribution.py your_dataset.json")
        print("3. Check the logs and model outputs in the specified directories")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()