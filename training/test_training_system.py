#!/usr/bin/env python3
"""
Simple test script to verify the speaker attribution training system.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add training module to path
training_dir = os.path.dirname(__file__)
booknlp_dir = os.path.dirname(training_dir)
sys.path.insert(0, booknlp_dir)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from training.config import TrainingConfig, DatasetConfig
        from training.data_utils import SpeakerAttributionDataProcessor
        print("✓ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("Testing configuration...")
    try:
        from training.config import TrainingConfig
        config = TrainingConfig()
        assert config.learning_rate == 2e-5
        assert config.batch_size == 16
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_data_processing():
    """Test data processing utilities."""
    print("Testing data processing...")
    try:
        from training.data_utils import SpeakerAttributionDataProcessor
        
        processor = SpeakerAttributionDataProcessor()
        
        # Test sample validation
        valid_sample = {
            "text": ["Hello", "world", "[QUOTE]", "Hi", "[/QUOTE]", "she", "said"],
            "quotes": [[2, 4]],
            "entities": [[0, 1, "PERSON", "Hello"], [5, 6, "PERSON", "she"]],
            "attributions": {"0": 1}
        }
        
        assert processor.validate_sample(valid_sample)
        
        # Test invalid sample
        invalid_sample = {
            "text": ["Hello"],
            "quotes": [[0, 5]],  # Out of bounds
            "entities": [],
            "attributions": {}
        }
        
        assert not processor.validate_sample(invalid_sample)
        
        print("✓ Data processing working")
        return True
    except Exception as e:
        print(f"✗ Data processing error: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation and saving."""
    print("Testing dataset creation...")
    try:
        from training.data_utils import SpeakerAttributionDataProcessor
        
        processor = SpeakerAttributionDataProcessor()
        
        # Create test data
        test_data = [
            {
                "text": ["John", "said", "[QUOTE]", "Hello", "[/QUOTE]"],
                "quotes": [[2, 4]],
                "entities": [[0, 1, "PERSON", "John"]],
                "attributions": {"0": 0}
            }
        ]
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            processor.save_dataset(test_data, temp_path)
            loaded_data = processor.load_dataset(temp_path)
            
            assert len(loaded_data) == 1
            assert loaded_data[0]["text"] == test_data[0]["text"]
            
            print("✓ Dataset creation and loading working")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ Dataset creation error: {e}")
        return False

def test_cli_validation():
    """Test CLI validation functionality."""
    print("Testing CLI validation...")
    try:
        # Create temporary dataset
        from training.data_utils import SpeakerAttributionDataProcessor
        
        processor = SpeakerAttributionDataProcessor()
        test_data = [
            {
                "text": ["Test", "quote", "[QUOTE]", "Hello", "[/QUOTE]"],
                "quotes": [[2, 4]],
                "entities": [[0, 1, "PERSON", "Test"]],
                "attributions": {"0": 0}
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            processor.save_dataset(test_data, temp_path)
            
            # Test validation function
            from training.train_speaker_attribution import validate_data_file
            result = validate_data_file(temp_path)
            assert result == True
            
            print("✓ CLI validation working")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ CLI validation error: {e}")
        return False

def test_gui_import():
    """Test GUI module import (may fail in headless environments)."""
    print("Testing GUI import...")
    try:
        # Try to import GUI module
        from training.dataset_creator_gui import DatasetCreatorGUI
        print("✓ GUI module imported successfully")
        return True
    except ImportError as e:
        if "tkinter" in str(e):
            print("⚠ GUI module requires tkinter (not available in headless environment)")
            return True  # This is expected in headless environments
        else:
            print(f"✗ GUI import error: {e}")
            return False
    except Exception as e:
        print(f"✗ GUI error: {e}")
        return False

def main():
    """Run all tests."""
    print("Speaker Attribution Training System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config, 
        test_data_processing,
        test_dataset_creation,
        test_cli_validation,
        test_gui_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())