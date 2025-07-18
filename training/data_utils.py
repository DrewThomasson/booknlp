"""
Data processing utilities for speaker attribution training.
"""

import json
import re
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class SpeakerAttributionDataProcessor:
    """Handles data processing for speaker attribution training."""
    
    def __init__(self, config=None):
        self.config = config
        self.quote_start_token = "[QUOTE]"
        self.quote_end_token = "[/QUOTE]"
        
    def tokenize_text_with_quotes(self, text: str, quotes: List[Tuple[int, int]]) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Tokenize text and insert quote markers.
        
        Args:
            text: Raw text string
            quotes: List of (start, end) character positions for quotes
            
        Returns:
            Tuple of (tokenized_text, adjusted_quote_positions)
        """
        # Simple whitespace tokenization for now
        tokens = text.split()
        
        # Convert character positions to token positions
        char_to_token = {}
        current_char = 0
        
        for token_idx, token in enumerate(tokens):
            # Find the token in the original text starting from current_char
            token_start = text.find(token, current_char)
            if token_start == -1:
                continue
                
            for char_idx in range(token_start, token_start + len(token)):
                char_to_token[char_idx] = token_idx
            current_char = token_start + len(token)
        
        # Convert quote positions to token positions
        token_quotes = []
        for quote_start, quote_end in quotes:
            token_start = char_to_token.get(quote_start)
            token_end = char_to_token.get(quote_end - 1)  # -1 because end is exclusive
            
            if token_start is not None and token_end is not None:
                token_quotes.append((token_start, token_end + 1))  # +1 to make end exclusive
        
        # Insert quote markers
        result_tokens = []
        adjusted_quotes = []
        offset = 0
        
        for quote_start, quote_end in sorted(token_quotes, reverse=True):
            # Insert end marker first (working backwards)
            tokens.insert(quote_end, self.quote_end_token)
            tokens.insert(quote_start, self.quote_start_token)
            
            # Adjust quote positions for the inserted tokens
            adjusted_quotes.append((quote_start + offset, quote_end + offset + 1))
            offset += 2
        
        adjusted_quotes.reverse()  # Restore original order
        return tokens, adjusted_quotes
    
    def load_dataset(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            raise
    
    def save_dataset(self, data: List[Dict[str, Any]], file_path: Union[str, Path]):
        """Save dataset to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} samples to {file_path}")
        except Exception as e:
            logger.error(f"Error saving dataset to {file_path}: {e}")
            raise
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a single data sample.
        
        Args:
            sample: Dictionary containing 'text', 'quotes', 'entities', 'attributions'
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = {'text', 'quotes', 'entities', 'attributions'}
        if not all(key in sample for key in required_keys):
            logger.warning(f"Sample missing required keys: {required_keys - set(sample.keys())}")
            return False
            
        text = sample['text']
        quotes = sample['quotes']
        entities = sample['entities']
        attributions = sample['attributions']
        
        # Check text is list of strings
        if not isinstance(text, list) or not all(isinstance(token, str) for token in text):
            logger.warning("Text must be a list of strings")
            return False
            
        # Check quotes format
        if not isinstance(quotes, list):
            logger.warning("Quotes must be a list")
            return False
            
        for quote in quotes:
            if not isinstance(quote, list) or len(quote) != 2:
                logger.warning("Each quote must be a list of [start, end]")
                return False
            start, end = quote
            if not (0 <= start < end <= len(text)):
                logger.warning(f"Quote indices {start}, {end} out of bounds for text length {len(text)}")
                return False
                
        # Check entities format  
        if not isinstance(entities, list):
            logger.warning("Entities must be a list")
            return False
            
        for entity in entities:
            if not isinstance(entity, list) or len(entity) != 4:
                logger.warning("Each entity must be a list of [start, end, type, text]")
                return False
            start, end, ent_type, ent_text = entity
            if not (0 <= start < end <= len(text)):
                logger.warning(f"Entity indices {start}, {end} out of bounds for text length {len(text)}")
                return False
                
        # Check attributions format
        if not isinstance(attributions, dict):
            logger.warning("Attributions must be a dictionary")
            return False
            
        for quote_idx, entity_idx in attributions.items():
            try:
                quote_idx = int(quote_idx)
                entity_idx = int(entity_idx)
            except ValueError:
                logger.warning("Attribution indices must be integers")
                return False
                
            if not (0 <= quote_idx < len(quotes)):
                logger.warning(f"Quote index {quote_idx} out of bounds")
                return False
            if not (0 <= entity_idx < len(entities)):
                logger.warning(f"Entity index {entity_idx} out of bounds")
                return False
                
        return True
    
    def prepare_training_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert samples to format expected by BERTSpeakerID model.
        
        Args:
            samples: List of validated samples
            
        Returns:
            Dictionary with training data in the format expected by the model
        """
        batch_x = []  # Text sequences
        batch_m = []  # Metadata (entity candidates, quotes)
        
        for sample in samples:
            text = sample['text']
            quotes = sample['quotes'] 
            entities = sample['entities']
            attributions = sample['attributions']
            
            # For each quote, create training examples
            for quote_idx, (quote_start, quote_end) in enumerate(quotes):
                
                # Get the correct entity for this quote (if any)
                correct_entity_idx = attributions.get(str(quote_idx), -1)
                
                # Prepare candidate entities
                candidates = []
                for ent_idx, (ent_start, ent_end, ent_type, ent_text) in enumerate(entities):
                    truth = 1 if ent_idx == correct_entity_idx else 0
                    candidates.append((ent_start, ent_end, truth, ent_idx))
                
                # Pad candidates to max_candidates (10)
                while len(candidates) < 10:
                    candidates.append((0, 0, 0, None))
                
                # Take only first 10 candidates
                candidates = candidates[:10]
                
                batch_x.append(text)
                batch_m.append((correct_entity_idx, candidates, (quote_start, quote_end)))
        
        return batch_x, batch_m
    
    def create_sample_from_text(self, text: str, entities: List[Dict], quotes: List[Dict]) -> Dict[str, Any]:
        """
        Create a training sample from raw text with detected entities and quotes.
        
        Args:
            text: Raw text string
            entities: List of entity dictionaries with keys: start, end, label, text
            quotes: List of quote dictionaries with keys: start, end, text
            
        Returns:
            Sample dictionary in the required format
        """
        # Convert character positions to token positions
        tokens = text.split()
        char_to_token = self._build_char_to_token_mapping(text, tokens)
        
        # Convert entities to required format
        processed_entities = []
        for ent in entities:
            token_start = char_to_token.get(ent['start'])
            token_end = char_to_token.get(ent['end'] - 1)
            
            if token_start is not None and token_end is not None:
                processed_entities.append([
                    token_start,
                    token_end + 1,  # Make end exclusive
                    ent['label'],
                    ent['text']
                ])
        
        # Convert quotes to required format  
        processed_quotes = []
        for quote in quotes:
            token_start = char_to_token.get(quote['start'])
            token_end = char_to_token.get(quote['end'] - 1)
            
            if token_start is not None and token_end is not None:
                processed_quotes.append([token_start, token_end + 1])
        
        # Insert quote markers
        tokens_with_quotes, adjusted_quotes = self.tokenize_text_with_quotes(text, [(q['start'], q['end']) for q in quotes])
        
        return {
            "text": tokens_with_quotes,
            "quotes": adjusted_quotes,
            "entities": processed_entities,
            "attributions": {}  # Will be filled by user in GUI
        }
    
    def _build_char_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """Build mapping from character positions to token indices."""
        char_to_token = {}
        current_char = 0
        
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, current_char)
            if token_start == -1:
                continue
                
            for char_idx in range(token_start, token_start + len(token)):
                char_to_token[char_idx] = token_idx
            current_char = token_start + len(token)
            
        return char_to_token
    
    def get_dataset_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not data:
            return {}
            
        total_samples = len(data)
        total_quotes = sum(len(sample['quotes']) for sample in data)
        total_entities = sum(len(sample['entities']) for sample in data)
        total_attributions = sum(len(sample['attributions']) for sample in data)
        
        entity_types = Counter()
        for sample in data:
            for entity in sample['entities']:
                entity_types[entity[2]] += 1
                
        avg_text_length = np.mean([len(sample['text']) for sample in data])
        
        return {
            'total_samples': total_samples,
            'total_quotes': total_quotes,
            'total_entities': total_entities,
            'total_attributions': total_attributions,
            'entity_types': dict(entity_types),
            'avg_text_length': avg_text_length,
            'attribution_ratio': total_attributions / total_quotes if total_quotes > 0 else 0
        }