#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
import os

def load_entities(file_path):
    """Load and parse the entities file"""
    return pd.read_csv(file_path, sep='\t')

def load_quotes(file_path):
    """Load and parse the quotes file"""
    return pd.read_csv(file_path, sep='\t')

def get_character_names(entities_df):
    """Get the best name for each character ID"""
    char_names = {}
    
    # Get all unique character IDs
    char_ids = entities_df['COREF'].unique()
    
    for char_id in char_ids:
        # Try to find a proper name (PROP) for this character
        prop_mentions = entities_df[(entities_df['COREF'] == char_id) & (entities_df['prop'] == 'PROP')]
        if not prop_mentions.empty:
            char_names[char_id] = prop_mentions.iloc[0]['text']
        else:
            # Fall back to the first mention
            first_mention = entities_df[entities_df['COREF'] == char_id]
            if not first_mention.empty and first_mention.iloc[0]['cat'] == 'PER':
                char_names[char_id] = first_mention.iloc[0]['text']
    
    return char_names

def create_character_network(entities_df, quotes_df, min_mentions=10):
    """Create a network of character interactions"""
    # Get character names
    char_names = get_character_names(entities_df)
    
    # Filter to only include characters (PER category)
    person_ids = set(entities_df[entities_df['cat'] == 'PER']['COREF'].unique())
    person_ids = {char_id for char_id in person_ids if char_id in char_names}
    
    # Count mentions for each character
    mention_counts = entities_df['COREF'].value_counts()
    
    # Filter to only include characters with significant presence
    main_chars = {char_id for char_id in person_ids if char_id in mention_counts.index and mention_counts[char_id] >= min_mentions}
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (characters)
    for char_id in main_chars:
        G.add_node(char_id, 
                  name=char_names.get(char_id, f"Character_{char_id}"),
                  weight=mention_counts.get(char_id, 0))
    
    # Add edges (interactions)
    # Process quotes to find character interactions
    last_speaker = None
    last_para = -1
    
    # Sort quotes by position in text
    sorted_quotes = quotes_df.sort_values(by='quote_start')
    
    for _, quote in sorted_quotes.iterrows():
        speaker_id = quote['char_id']
        
        # Skip if speaker is not a main character
        if speaker_id not in main_chars:
            continue
        
        # Check if this is a conversation with the previous speaker
        if last_speaker is not None and last_speaker != speaker_id:
            if G.has_edge(speaker_id, last_speaker):
                # Increment weight if edge exists
                G[speaker_id][last_speaker]['weight'] += 1
            else:
                # Create new edge
                G.add_edge(speaker_id, last_speaker, weight=1)
        
        last_speaker = speaker_id
    
    return G, char_names

def plot_character_network(G, char_names, output_path=None, title=None):
    """Plot the character network"""
    plt.figure(figsize=(14, 10))
    
    # Get node sizes based on character mentions
    node_sizes = [G.nodes[n]['weight'] * 5 for n in G.nodes()]
    
    # Get edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w/max_edge_weight * 5 for w in edge_weights]
    
    # Set position layout
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # Draw the network
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # Add labels with character names
    labels = {node: char_names.get(node, f"Character_{node}") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    if title:
        plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Character network saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create character network from BookNLP output')
    parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    parser.add_argument('--min-mentions', type=int, default=10, help='Minimum mentions for a character to be included')
    parser.add_argument('--output', '-o', help='Output file path for the network visualization')
    parser.add_argument('--title', help='Title for the network visualization')
    
    args = parser.parse_args()
    
    book_path = Path(args.book_dir)
    book_name = book_path.name.split('.')[0]
    
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
    
    entities_df = load_entities(entities_file)
    quotes_df = load_quotes(quotes_file)
    
    # Create network
    G, char_names = create_character_network(entities_df, quotes_df, args.min_mentions)
    
    # Plot network
    title = args.title if args.title else f"Character Network: {book_name}"
    plot_character_network(G, char_names, args.output, title)
    
    # Print some statistics
    print(f"Number of characters: {G.number_of_nodes()}")
    print(f"Number of interactions: {G.number_of_edges()}")
    
    # Find central characters
    centrality = nx.degree_centrality(G)
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost central characters:")
    for char_id, score in sorted_centrality[:5]:
        name = char_names.get(char_id, f"Character_{char_id}")
        print(f"{name}: {score:.3f}")

if __name__ == '__main__':
    main() 