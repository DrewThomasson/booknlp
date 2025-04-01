#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import networkx as nx
import json
from jinja2 import Template
import analyze_booknlp
import character_network

def generate_html_dashboard(book_dir, output_dir):
    """Generate an HTML dashboard for BookNLP output"""
    book_path = Path(book_dir)
    book_name = book_path.name.split('.')[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
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
    
    entities_df = pd.read_csv(entities_file, sep='\t')
    quotes_df = pd.read_csv(quotes_file, sep='\t')
    tokens_df = pd.read_csv(tokens_file, sep='\t')
    
    # Create character stats
    top_chars = analyze_booknlp.top_characters(entities_df, n=20)
    character_types = analyze_booknlp.character_types(entities_df)
    quotes_dist = analyze_booknlp.quote_distribution(quotes_df, entities_df)
    
    # Generate figures
    char_mentions_path = os.path.join(figures_dir, f"{book_name}_character_mentions.png")
    analyze_booknlp.plot_character_mentions(top_chars.head(10), char_mentions_path)
    
    quote_dist_path = os.path.join(figures_dir, f"{book_name}_quote_distribution.png")
    analyze_booknlp.plot_quote_distribution(quotes_dist.head(10), quote_dist_path)
    
    network_path = os.path.join(figures_dir, f"{book_name}_character_network.png")
    G, char_names = character_network.create_character_network(entities_df, quotes_df, min_mentions=10)
    character_network.plot_character_network(G, char_names, network_path, title=f"Character Network: {book_name}")
    
    # Calculate centrality
    centrality = nx.degree_centrality(G)
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    central_chars = []
    for char_id, score in sorted_centrality[:5]:
        name = char_names.get(char_id, f"Character_{char_id}")
        central_chars.append({"name": name, "score": score})
    
    # Get sample quotes for top characters
    sample_quotes = {}
    for char_id in top_chars['char_id'].head(5):
        char_quotes = quotes_df[quotes_df['char_id'] == char_id]
        if not char_quotes.empty:
            # Get a representative quote (choosing the longest among the first 10)
            sample_quotes[char_id] = sorted(char_quotes.head(10)['quote'].tolist(), key=len, reverse=True)[0]
    
    # Get character names for the quotes
    for char_id in sample_quotes:
        char_row = top_chars[top_chars['char_id'] == char_id]
        if not char_row.empty:
            char_name = char_row.iloc[0]['character_name']
            sample_quotes[char_id] = {
                "name": char_name,
                "quote": sample_quotes[char_id]
            }
    
    # Create HTML template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ book_name }} - BookNLP Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .section {
                margin-bottom: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            .figure {
                text-align: center;
                margin: 20px 0;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .quote {
                font-style: italic;
                background: #f1f1f1;
                padding: 15px;
                border-left: 5px solid #2c3e50;
                margin: 10px 0;
            }
            .stats {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            .stat-box {
                flex: 0 0 30%;
                background: #fff;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
        </style>
    </head>
    <body>
        <h1>{{ book_name }} - BookNLP Analysis</h1>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Characters</h3>
                    <p>Total characters: {{ entities_df['COREF'].nunique() }}</p>
                    <p>People: {{ character_types.get('PER', 0) }}</p>
                    <p>Locations: {{ character_types.get('LOC', 0) }}</p>
                    <p>Organizations: {{ character_types.get('ORG', 0) }}</p>
                </div>
                <div class="stat-box">
                    <h3>Dialogue</h3>
                    <p>Total quotes: {{ quotes_df.shape[0] }}</p>
                    <p>Characters speaking: {{ quotes_df['char_id'].nunique() }}</p>
                </div>
                <div class="stat-box">
                    <h3>Text</h3>
                    <p>Total tokens: {{ tokens_df.shape[0] }}</p>
                    <p>Total sentences: {{ tokens_df['sentence_ID'].nunique() }}</p>
                    <p>Total paragraphs: {{ tokens_df['paragraph_ID'].nunique() }}</p>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="section">
                <h2>Top Characters</h2>
                <table>
                    <tr>
                        <th>Character</th>
                        <th>Mentions</th>
                    </tr>
                    {% for _, row in top_chars.head(10).iterrows() %}
                    <tr>
                        <td>{{ row['character_name'] }}</td>
                        <td>{{ row['mention_count'] }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="figure">
                    <img src="figures/{{ book_name }}_character_mentions.png" alt="Character Mentions">
                </div>
            </div>
            
            <div class="section">
                <h2>Quote Distribution</h2>
                <table>
                    <tr>
                        <th>Character</th>
                        <th>Quotes</th>
                    </tr>
                    {% for _, row in quotes_dist.head(10).iterrows() %}
                    <tr>
                        <td>{{ row['character_name'] }}</td>
                        <td>{{ row['quote_count'] }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="figure">
                    <img src="figures/{{ book_name }}_quote_distribution.png" alt="Quote Distribution">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Character Network</h2>
            <div class="figure">
                <img src="figures/{{ book_name }}_character_network.png" alt="Character Network">
            </div>
            
            <h3>Most Central Characters</h3>
            <table>
                <tr>
                    <th>Character</th>
                    <th>Centrality Score</th>
                </tr>
                {% for char in central_chars %}
                <tr>
                    <td>{{ char['name'] }}</td>
                    <td>{{ "%.3f"|format(char['score']) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="section">
            <h2>Sample Quotes</h2>
            {% for char_id, quote_data in sample_quotes.items() %}
            <h3>{{ quote_data['name'] }}</h3>
            <div class="quote">
                {{ quote_data['quote'] }}
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    
    # Render template
    template = Template(template_str)
    html = template.render(
        book_name=book_name,
        entities_df=entities_df,
        quotes_df=quotes_df,
        tokens_df=tokens_df,
        top_chars=top_chars,
        character_types=character_types,
        quotes_dist=quotes_dist,
        central_chars=central_chars,
        sample_quotes=sample_quotes
    )
    
    # Write HTML file
    html_path = os.path.join(output_dir, f"{book_name}_dashboard.html")
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"Dashboard generated at {html_path}")
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Generate HTML dashboard for BookNLP output')
    parser.add_argument('book_dir', help='Directory containing BookNLP output files')
    parser.add_argument('--output', '-o', default='dashboard', help='Output directory for dashboard')
    
    args = parser.parse_args()
    
    html_path = generate_html_dashboard(args.book_dir, args.output)
    
    print(f"Open the dashboard at: file://{os.path.abspath(html_path)}")

if __name__ == '__main__':
    main() 