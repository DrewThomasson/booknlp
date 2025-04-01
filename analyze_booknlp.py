#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_entities(file_path):
    """Load and parse the entities file"""
    return pd.read_csv(file_path, sep='\t')

def load_quotes(file_path):
    """Load and parse the quotes file"""
    return pd.read_csv(file_path, sep='\t')

def load_tokens(file_path):
    """Load and parse the tokens file"""
    return pd.read_csv(file_path, sep='\t')

def top_characters(entities_df, n=10):
    """Find top n characters by number of mentions"""
    # Group by character ID and count occurrences
    char_counts = entities_df['COREF'].value_counts().reset_index()
    char_counts.columns = ['char_id', 'mention_count']
    
    # Get character names (using the first PROP mention if available, otherwise first mention)
    char_names = {}
    for char_id in char_counts['char_id']:
        # Try to find a proper name (PROP) for this character
        prop_mentions = entities_df[(entities_df['COREF'] == char_id) & (entities_df['prop'] == 'PROP')]
        if not prop_mentions.empty:
            char_names[char_id] = prop_mentions.iloc[0]['text']
        else:
            # Fall back to the first mention
            char_names[char_id] = entities_df[entities_df['COREF'] == char_id].iloc[0]['text']
    
    # Add character names to the dataframe
    char_counts['character_name'] = char_counts['char_id'].map(char_names)
    
    return char_counts.head(n)

def character_types(entities_df):
    """Analyze the types of characters (PER, LOC, etc.)"""
    return entities_df['cat'].value_counts()

def quote_distribution(quotes_df, entities_df):
    """Analyze the distribution of quotes among characters"""
    # Count quotes by character
    quote_counts = quotes_df['char_id'].value_counts().reset_index()
    quote_counts.columns = ['char_id', 'quote_count']
    
    # Get character names
    char_names = {}
    for char_id in quote_counts['char_id']:
        # Try to find a proper name (PROP) for this character
        prop_mentions = entities_df[(entities_df['COREF'] == char_id) & (entities_df['prop'] == 'PROP')]
        if not prop_mentions.empty:
            char_names[char_id] = prop_mentions.iloc[0]['text']
        else:
            # Fall back to the first mention
            first_mention = entities_df[entities_df['COREF'] == char_id]
            if not first_mention.empty:
                char_names[char_id] = first_mention.iloc[0]['text']
            else:
                char_names[char_id] = f"Character_{char_id}"
    
    # Add character names to the dataframe
    quote_counts['character_name'] = quote_counts['char_id'].map(char_names)
    
    return quote_counts

def plot_character_mentions(char_counts, output_path=None):
    """Plot the number of mentions for each main character"""
    plt.figure(figsize=(12, 6))
    plt.bar(char_counts['character_name'], char_counts['mention_count'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Character Mentions')
    plt.ylabel('Number of Mentions')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Character mentions plot saved to {output_path}")
    else:
        plt.show()

def plot_quote_distribution(quote_counts, output_path=None):
    """Plot the distribution of quotes among characters"""
    plt.figure(figsize=(12, 6))
    plt.bar(quote_counts['character_name'], quote_counts['quote_count'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Character Quote Distribution')
    plt.ylabel('Number of Quotes')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Quote distribution plot saved to {output_path}")
    else:
        plt.show()

def analyze_book(book_dir, output_dir=None):
    """Analyze BookNLP output for a single book"""
    book_path = Path(book_dir)
    book_name = book_path.name.split('.')[0]
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find the actual file names in the directory
    file_prefix = None
    for file in os.listdir(book_path):
        if file.endswith('.entities'):
            file_prefix = file.split('.entities')[0]
            break
    
    if not file_prefix:
        raise FileNotFoundError(f"Could not find .entities file in {book_path}")
    
    # Load data
    entities_file = book_path / f"{file_prefix}.entities"
    quotes_file = book_path / f"{file_prefix}.quotes"
    tokens_file = book_path / f"{file_prefix}.tokens"
    
    print(f"Analyzing book: {book_name}")
    
    entities_df = load_entities(entities_file)
    quotes_df = load_quotes(quotes_file)
    tokens_df = load_tokens(tokens_file)
    
    # Analyze top characters
    top_chars = top_characters(entities_df)
    print("\nTop 10 characters by mentions:")
    print(top_chars[['character_name', 'mention_count']])
    
    # Analyze character types
    char_types = character_types(entities_df)
    print("\nCharacter types:")
    print(char_types)
    
    # Analyze quote distribution
    quotes_dist = quote_distribution(quotes_df, entities_df)
    print("\nQuote distribution among top characters:")
    print(quotes_dist.head(10)[['character_name', 'quote_count']])
    
    # Create plots
    if output_dir:
        plot_character_mentions(top_chars, os.path.join(output_dir, f"{book_name}_character_mentions.png"))
        plot_quote_distribution(quotes_dist.head(10), os.path.join(output_dir, f"{book_name}_quote_distribution.png"))
    else:
        plot_character_mentions(top_chars)
        plot_quote_distribution(quotes_dist.head(10))
    
    return {
        'top_characters': top_chars,
        'character_types': char_types,
        'quote_distribution': quotes_dist
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze BookNLP output files')
    parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    parser.add_argument('--output', '-o', help='Directory to save output files and plots')
    
    args = parser.parse_args()
    
    analyze_book(args.book_dir, args.output)

if __name__ == '__main__':
    main() 