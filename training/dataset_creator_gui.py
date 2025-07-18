"""
Interactive GUI for creating speaker attribution training datasets.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# For BookNLP integration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatasetConfig
from data_utils import SpeakerAttributionDataProcessor

logger = logging.getLogger(__name__)

class DatasetCreatorGUI:
    """GUI application for creating speaker attribution datasets."""
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self.processor = SpeakerAttributionDataProcessor()
        
        # Data storage
        self.current_text = ""
        self.current_tokens = []
        self.detected_entities = []
        self.detected_quotes = []
        self.current_dataset = []
        self.current_sample_index = -1
        
        # GUI state
        self.selected_quote_idx = None
        self.selected_entity_idx = None
        
        # Initialize GUI
        self.root = tk.Tk()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI interface."""
        self.root.title("Speaker Attribution Dataset Creator")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        
        # Create main menu
        self.create_menu()
        
        # Create main layout
        self.create_main_layout()
        
    def create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Text File", command=self.load_text_file)
        file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Save Dataset", command=self.save_dataset)
        file_menu.add_command(label="Save Dataset As...", command=self.save_dataset_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Run Entity Detection", command=self.run_entity_detection)
        edit_menu.add_command(label="Run Quote Detection", command=self.run_quote_detection)
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create the main application layout."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Text and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)
        
        # Right panel - Entities and quotes
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_left_panel(self, parent):
        """Create the left panel with text editor and controls."""
        # Text editor frame
        text_frame = ttk.LabelFrame(parent, text="Text Content")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text widget with scrollbar
        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=(self.config.font_family, self.config.font_size),
            state=tk.DISABLED
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Text", command=self.load_text_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Detect Entities", command=self.run_entity_detection).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Detect Quotes", command=self.run_quote_detection).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Create Sample", command=self.create_sample).pack(side=tk.LEFT, padx=2)
        
    def create_right_panel(self, parent):
        """Create the right panel with entities, quotes, and dataset management."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Entities tab
        entities_frame = ttk.Frame(notebook)
        notebook.add(entities_frame, text="Entities")
        self.create_entities_tab(entities_frame)
        
        # Quotes tab
        quotes_frame = ttk.Frame(notebook)
        notebook.add(quotes_frame, text="Quotes")
        self.create_quotes_tab(quotes_frame)
        
        # Dataset tab
        dataset_frame = ttk.Frame(notebook)
        notebook.add(dataset_frame, text="Dataset")
        self.create_dataset_tab(dataset_frame)
    
    def create_entities_tab(self, parent):
        """Create the entities management tab."""
        # Entities list
        entities_label = ttk.Label(parent, text="Detected Entities:")
        entities_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Create treeview for entities
        self.entities_tree = ttk.Treeview(parent, columns=('Type', 'Text'), show='tree headings', height=8)
        self.entities_tree.heading('#0', text='Index')
        self.entities_tree.heading('Type', text='Type')
        self.entities_tree.heading('Text', text='Text')
        self.entities_tree.column('#0', width=50)
        self.entities_tree.column('Type', width=80)
        self.entities_tree.column('Text', width=150)
        
        # Scrollbar for entities tree
        entities_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.entities_tree.yview)
        self.entities_tree.configure(yscrollcommand=entities_scrollbar.set)
        
        # Pack entities tree and scrollbar
        entities_tree_frame = ttk.Frame(parent)
        entities_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.entities_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        entities_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind entity selection
        self.entities_tree.bind('<<TreeviewSelect>>', self.on_entity_select)
    
    def create_quotes_tab(self, parent):
        """Create the quotes management tab."""
        # Quotes list
        quotes_label = ttk.Label(parent, text="Detected Quotes:")
        quotes_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Create treeview for quotes
        self.quotes_tree = ttk.Treeview(parent, columns=('Text',), show='tree headings', height=8)
        self.quotes_tree.heading('#0', text='Index')
        self.quotes_tree.heading('Text', text='Quote Text')
        self.quotes_tree.column('#0', width=50)
        self.quotes_tree.column('Text', width=250)
        
        # Scrollbar for quotes tree
        quotes_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.quotes_tree.yview)
        self.quotes_tree.configure(yscrollcommand=quotes_scrollbar.set)
        
        # Pack quotes tree and scrollbar
        quotes_tree_frame = ttk.Frame(parent)
        quotes_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.quotes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        quotes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Attribution controls
        attr_frame = ttk.LabelFrame(parent, text="Attribution")
        attr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(attr_frame, text="Selected Quote:").pack(anchor=tk.W, padx=5)
        self.selected_quote_var = tk.StringVar(value="None")
        ttk.Label(attr_frame, textvariable=self.selected_quote_var, foreground="blue").pack(anchor=tk.W, padx=5)
        
        ttk.Label(attr_frame, text="Selected Entity:").pack(anchor=tk.W, padx=5)
        self.selected_entity_var = tk.StringVar(value="None")
        ttk.Label(attr_frame, textvariable=self.selected_entity_var, foreground="green").pack(anchor=tk.W, padx=5)
        
        ttk.Button(attr_frame, text="Create Attribution", command=self.create_attribution).pack(pady=5)
        
        # Bind quote selection
        self.quotes_tree.bind('<<TreeviewSelect>>', self.on_quote_select)
    
    def create_dataset_tab(self, parent):
        """Create the dataset management tab."""
        # Dataset info
        info_frame = ttk.LabelFrame(parent, text="Dataset Info")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.dataset_info_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(info_frame, textvariable=self.dataset_info_var).pack(padx=5, pady=5)
        
        # Dataset samples list
        samples_frame = ttk.LabelFrame(parent, text="Samples")
        samples_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.samples_tree = ttk.Treeview(samples_frame, columns=('Quotes', 'Entities', 'Attributions'), show='tree headings')
        self.samples_tree.heading('#0', text='Index')
        self.samples_tree.heading('Quotes', text='Quotes')
        self.samples_tree.heading('Entities', text='Entities')
        self.samples_tree.heading('Attributions', text='Attributions')
        self.samples_tree.column('#0', width=50)
        self.samples_tree.column('Quotes', width=60)
        self.samples_tree.column('Entities', width=60)
        self.samples_tree.column('Attributions', width=80)
        
        self.samples_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Dataset controls
        dataset_controls = ttk.Frame(parent)
        dataset_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(dataset_controls, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(dataset_controls, text="Save Dataset", command=self.save_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(dataset_controls, text="Clear Dataset", command=self.clear_dataset).pack(side=tk.LEFT, padx=2)
    
    def load_text_file(self):
        """Load text file for processing."""
        file_path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.current_text = f.read()
                
                # Update text widget
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, self.current_text)
                self.text_widget.config(state=tk.DISABLED)
                
                # Clear previous detections
                self.detected_entities = []
                self.detected_quotes = []
                self.update_entities_display()
                self.update_quotes_display()
                
                self.status_var.set(f"Loaded text file: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load text file: {e}")
    
    def run_entity_detection(self):
        """Run entity detection using BookNLP."""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a text file first.")
            return
        
        try:
            self.status_var.set("Running entity detection...")
            self.root.update()
            
            # Simple entity detection placeholder
            # In a real implementation, this would use the BookNLP pipeline
            self.detected_entities = self._mock_entity_detection(self.current_text)
            
            self.update_entities_display()
            self.status_var.set(f"Detected {len(self.detected_entities)} entities")
            
        except Exception as e:
            messagebox.showerror("Error", f"Entity detection failed: {e}")
            self.status_var.set("Entity detection failed")
    
    def run_quote_detection(self):
        """Run quote detection."""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a text file first.")
            return
        
        try:
            self.status_var.set("Running quote detection...")
            self.root.update()
            
            # Simple quote detection placeholder
            self.detected_quotes = self._mock_quote_detection(self.current_text)
            
            self.update_quotes_display()
            self.status_var.set(f"Detected {len(self.detected_quotes)} quotes")
            
        except Exception as e:
            messagebox.showerror("Error", f"Quote detection failed: {e}")
            self.status_var.set("Quote detection failed")
    
    def _mock_entity_detection(self, text: str) -> List[Dict]:
        """Mock entity detection for demonstration."""
        # Simple regex-based entity detection
        import re
        
        entities = []
        # Look for capitalized words (simple person detection)
        for match in re.finditer(r'\b[A-Z][a-z]+\b', text):
            if match.group() not in ['The', 'This', 'That', 'And', 'But', 'He', 'She', 'It']:
                entities.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'PERSON',
                    'text': match.group()
                })
        
        return entities[:20]  # Limit to first 20 for demo
    
    def _mock_quote_detection(self, text: str) -> List[Dict]:
        """Mock quote detection for demonstration."""
        import re
        
        quotes = []
        # Simple quote detection
        for match in re.finditer(r'"([^"]+)"', text):
            quotes.append({
                'start': match.start() + 1,  # Skip opening quote
                'end': match.end() - 1,     # Skip closing quote
                'text': match.group(1)
            })
        
        return quotes[:10]  # Limit to first 10 for demo
    
    def update_entities_display(self):
        """Update the entities tree view."""
        # Clear existing items
        for item in self.entities_tree.get_children():
            self.entities_tree.delete(item)
        
        # Add detected entities
        for i, entity in enumerate(self.detected_entities):
            self.entities_tree.insert('', 'end', iid=str(i), text=str(i), 
                                    values=(entity['label'], entity['text']))
    
    def update_quotes_display(self):
        """Update the quotes tree view."""
        # Clear existing items
        for item in self.quotes_tree.get_children():
            self.quotes_tree.delete(item)
        
        # Add detected quotes
        for i, quote in enumerate(self.detected_quotes):
            text_preview = quote['text'][:50] + "..." if len(quote['text']) > 50 else quote['text']
            self.quotes_tree.insert('', 'end', iid=str(i), text=str(i), 
                                   values=(text_preview,))
    
    def on_entity_select(self, event):
        """Handle entity selection."""
        selection = self.entities_tree.selection()
        if selection:
            entity_idx = int(selection[0])
            entity = self.detected_entities[entity_idx]
            self.selected_entity_idx = entity_idx
            self.selected_entity_var.set(f"{entity_idx}: {entity['text']} ({entity['label']})")
    
    def on_quote_select(self, event):
        """Handle quote selection."""
        selection = self.quotes_tree.selection()
        if selection:
            quote_idx = int(selection[0])
            quote = self.detected_quotes[quote_idx]
            self.selected_quote_idx = quote_idx
            quote_preview = quote['text'][:30] + "..." if len(quote['text']) > 30 else quote['text']
            self.selected_quote_var.set(f"{quote_idx}: {quote_preview}")
    
    def create_attribution(self):
        """Create an attribution between selected quote and entity."""
        if self.selected_quote_idx is None:
            messagebox.showwarning("Warning", "Please select a quote first.")
            return
        
        if self.selected_entity_idx is None:
            messagebox.showwarning("Warning", "Please select an entity first.")
            return
        
        # This would be used when creating the actual sample
        messagebox.showinfo("Success", 
                           f"Attribution created: Quote {self.selected_quote_idx} -> Entity {self.selected_entity_idx}")
    
    def create_sample(self):
        """Create a training sample from current text, entities, and quotes."""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load text first.")
            return
        
        if not self.detected_entities:
            messagebox.showwarning("Warning", "Please detect entities first.")
            return
        
        if not self.detected_quotes:
            messagebox.showwarning("Warning", "Please detect quotes first.")
            return
        
        try:
            # Create sample using data processor
            sample = self.processor.create_sample_from_text(
                self.current_text, self.detected_entities, self.detected_quotes
            )
            
            # Add to dataset
            self.current_dataset.append(sample)
            self.update_dataset_display()
            
            messagebox.showinfo("Success", f"Sample created! Dataset now has {len(self.current_dataset)} samples.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample: {e}")
    
    def load_dataset(self):
        """Load existing dataset."""
        file_path = filedialog.askopenfilename(
            title="Load dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_dataset = self.processor.load_dataset(file_path)
                self.update_dataset_display()
                messagebox.showinfo("Success", f"Loaded dataset with {len(self.current_dataset)} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
    
    def save_dataset(self):
        """Save current dataset."""
        if not self.current_dataset:
            messagebox.showwarning("Warning", "No dataset to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save dataset",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.processor.save_dataset(self.current_dataset, file_path)
                messagebox.showinfo("Success", f"Dataset saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {e}")
    
    def save_dataset_as(self):
        """Save dataset with new filename."""
        self.save_dataset()
    
    def clear_dataset(self):
        """Clear current dataset."""
        if messagebox.askyesno("Confirm", "Clear all samples from dataset?"):
            self.current_dataset = []
            self.update_dataset_display()
    
    def clear_all(self):
        """Clear all data."""
        if messagebox.askyesno("Confirm", "Clear all data (text, entities, quotes, dataset)?"):
            self.current_text = ""
            self.detected_entities = []
            self.detected_quotes = []
            self.current_dataset = []
            
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.config(state=tk.DISABLED)
            
            self.update_entities_display()
            self.update_quotes_display()
            self.update_dataset_display()
            
            self.status_var.set("All data cleared")
    
    def update_dataset_display(self):
        """Update dataset samples display."""
        # Clear existing items
        for item in self.samples_tree.get_children():
            self.samples_tree.delete(item)
        
        # Add samples
        for i, sample in enumerate(self.current_dataset):
            num_quotes = len(sample.get('quotes', []))
            num_entities = len(sample.get('entities', []))
            num_attributions = len(sample.get('attributions', {}))
            
            self.samples_tree.insert('', 'end', iid=str(i), text=str(i),
                                   values=(num_quotes, num_entities, num_attributions))
        
        # Update info
        if self.current_dataset:
            stats = self.processor.get_dataset_statistics(self.current_dataset)
            info_text = f"Samples: {stats.get('total_samples', 0)}, " \
                       f"Quotes: {stats.get('total_quotes', 0)}, " \
                       f"Entities: {stats.get('total_entities', 0)}, " \
                       f"Attributions: {stats.get('total_attributions', 0)}"
            self.dataset_info_var.set(info_text)
        else:
            self.dataset_info_var.set("No dataset loaded")
    
    def show_instructions(self):
        """Show usage instructions."""
        instructions = """
Speaker Attribution Dataset Creator - Instructions

1. Load Text File: Click 'Load Text' to load a text file for processing.

2. Entity Detection: Click 'Detect Entities' to automatically detect person entities in the text.

3. Quote Detection: Click 'Detect Quotes' to automatically detect quoted speech in the text.

4. Create Attributions: 
   - Select a quote from the Quotes tab
   - Select an entity from the Entities tab  
   - Click 'Create Attribution' to link them

5. Create Sample: Click 'Create Sample' to add the current text with its entities, quotes, and attributions to the dataset.

6. Manage Dataset: Use the Dataset tab to view, load, and save your training dataset.

7. Save Dataset: Save your dataset as a JSON file for use in training.

The created dataset will be in the format required for speaker attribution model training.
        """
        
        messagebox.showinfo("Instructions", instructions)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
Speaker Attribution Dataset Creator
Version 1.0

A GUI tool for creating training datasets for speaker attribution models.
Part of the BookNLP training system.

Created for use with the BERTSpeakerID model.
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Start the GUI application."""
        logger.info("Starting Dataset Creator GUI")
        self.root.mainloop()

def main():
    """Main entry point for the GUI application."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run GUI
    config = DatasetConfig()
    app = DatasetCreatorGUI(config)
    app.run()

if __name__ == "__main__":
    main()