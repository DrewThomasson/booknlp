#!/usr/bin/env python3
import argparse
import os
from booknlp.booknlp import BookNLP

def main():
    parser = argparse.ArgumentParser(description='Run BookNLP pipeline on a text file')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--model', choices=['small', 'big'], default='small', help='Model size to use')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract book_id from input filename
    book_id = args.input_file.split('/')[-1].split('.')[0]

    # Model parameters with BERT enabled
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",  # Run all pipeline components
        "model": args.model,  # Use specified model size
        "use_bert": True,  # Enable BERT for better accuracy
        "bert_model": "bert-base-uncased"  # Use base BERT model
    }

    # Initialize BookNLP with English language and model parameters
    booknlp = BookNLP("en", model_params)
    
    # Process the input file and save results to output directory
    print(f"Processing {args.input_file}...")
    booknlp.process(args.input_file, args.output_dir, book_id)
    print(f"Results saved in {args.output_dir}/{book_id}.*")

if __name__ == "__main__":
    main() 