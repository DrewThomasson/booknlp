#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import analyze_booknlp
import character_network
import generate_dashboard

def find_file_prefix(book_dir):
    """Find the file prefix used in the BookNLP output directory"""
    book_path = Path(book_dir)
    file_prefix = None
    for file in os.listdir(book_path):
        if file.endswith('.entities'):
            file_prefix = file.split('.entities')[0]
            break
    
    if not file_prefix:
        raise FileNotFoundError(f"Could not find .entities file in {book_path}")
    
    return file_prefix

def main():
    parser = argparse.ArgumentParser(
        description='BookNLP Analysis Tools - Visualize and analyze BookNLP output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a full dashboard for a book
  python booknlp_analyzer.py dashboard output/pride_improved
  
  # Plot character mentions
  python booknlp_analyzer.py character-stats output/pride_improved
  
  # Create a character network visualization
  python booknlp_analyzer.py character-network output/pride_improved
  
  # Analyze all books in the output directory
  python booknlp_analyzer.py analyze-all output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate an HTML dashboard with all analyses')
    dashboard_parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    dashboard_parser.add_argument('--output', '-o', default='dashboard', help='Output directory for dashboard')
    
    # Character stats command
    stats_parser = subparsers.add_parser('character-stats', help='Generate character statistics and visualizations')
    stats_parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    stats_parser.add_argument('--output', '-o', help='Output directory for visualizations')
    
    # Character network command
    network_parser = subparsers.add_parser('character-network', help='Generate character network visualization')
    network_parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    network_parser.add_argument('--min-mentions', type=int, default=10, help='Minimum mentions for a character to be included')
    network_parser.add_argument('--output', '-o', help='Output file path for the network visualization')
    
    # Analyze all books command
    all_parser = subparsers.add_parser('analyze-all', help='Analyze all books in a directory')
    all_parser.add_argument('directory', help='Directory containing multiple BookNLP output directories')
    all_parser.add_argument('--output', '-o', default='dashboard', help='Output directory for dashboards')
    
    args = parser.parse_args()
    
    if args.command == 'dashboard':
        html_path = generate_dashboard.generate_html_dashboard(args.book_dir, args.output)
        print(f"Open the dashboard at: file://{os.path.abspath(html_path)}")
    
    elif args.command == 'character-stats':
        book_path = Path(args.book_dir)
        book_name = book_path.name.split('.')[0]
        file_prefix = find_file_prefix(args.book_dir)
        
        # Load data
        entities_file = book_path / f"{file_prefix}.entities"
        quotes_file = book_path / f"{file_prefix}.quotes"
        
        # Process data
        entities_df = analyze_booknlp.load_entities(entities_file)
        quotes_df = analyze_booknlp.load_quotes(quotes_file)
        
        # Generate analyses
        top_chars = analyze_booknlp.top_characters(entities_df)
        char_types = analyze_booknlp.character_types(entities_df)
        quotes_dist = analyze_booknlp.quote_distribution(quotes_df, entities_df)
        
        # Print results
        print(f"\nAnalyzing book: {book_name}")
        print("\nTop 10 characters by mentions:")
        print(top_chars[['character_name', 'mention_count']])
        
        print("\nCharacter types:")
        print(char_types)
        
        print("\nQuote distribution among top characters:")
        print(quotes_dist.head(10)[['character_name', 'quote_count']])
        
        # Create visualizations
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            analyze_booknlp.plot_character_mentions(
                top_chars, 
                os.path.join(args.output, f"{book_name}_character_mentions.png")
            )
            analyze_booknlp.plot_quote_distribution(
                quotes_dist.head(10), 
                os.path.join(args.output, f"{book_name}_quote_distribution.png")
            )
        else:
            print("\nCharacter Mentions Plot:")
            analyze_booknlp.plot_character_mentions(top_chars)
            
            print("\nQuote Distribution Plot:")
            analyze_booknlp.plot_quote_distribution(quotes_dist.head(10))
    
    elif args.command == 'character-network':
        book_path = Path(args.book_dir)
        book_name = book_path.name.split('.')[0]
        file_prefix = find_file_prefix(args.book_dir)
        
        # Load data
        entities_file = book_path / f"{file_prefix}.entities"
        quotes_file = book_path / f"{file_prefix}.quotes"
        
        entities_df = analyze_booknlp.load_entities(entities_file)
        quotes_df = analyze_booknlp.load_quotes(quotes_file)
        
        # Create network
        G, char_names = character_network.create_character_network(entities_df, quotes_df, args.min_mentions)
        
        # Plot network
        output_path = args.output if args.output else None
        character_network.plot_character_network(
            G, 
            char_names, 
            output_path, 
            title=f"Character Network: {book_name}"
        )
        
        # Print some statistics
        print(f"Character network for: {book_name}")
        print(f"Number of characters: {G.number_of_nodes()}")
        print(f"Number of interactions: {G.number_of_edges()}")
        
        # Find central characters
        import networkx as nx
        centrality = nx.degree_centrality(G)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost central characters:")
        for char_id, score in sorted_centrality[:5]:
            name = char_names.get(char_id, f"Character_{char_id}")
            print(f"{name}: {score:.3f}")
    
    elif args.command == 'analyze-all':
        # Find all book directories
        dir_path = Path(args.directory)
        book_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
        
        if not book_dirs:
            print(f"No book directories found in {args.directory}")
            return
        
        print(f"Found {len(book_dirs)} book directories to analyze")
        
        # Process each book
        for book_dir in book_dirs:
            try:
                html_path = generate_dashboard.generate_html_dashboard(book_dir, args.output)
                print(f"Generated dashboard for {book_dir.name}: {html_path}")
            except Exception as e:
                print(f"Error processing {book_dir.name}: {e}")
        
        print(f"\nAll dashboards have been generated in: {os.path.abspath(args.output)}")
        print("You can open them in your web browser to view the analyses.")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 