#!/usr/bin/env python3
"""
Update Baby Stroller Graphs with Specific Order and Colors
Updates both temperature 0.0 and 0.3 graphs with exact model order and colors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the exact order and colors
MODEL_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106", 
    "gpt-3.5-turbo-instruct",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-0613",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest"
]

# Define colors for each model
MODEL_COLORS = {
    "gpt-3.5-turbo-0125": "#d62728",      # muted red
    "gpt-3.5-turbo-1106": "#d62728",      # muted red
    "gpt-3.5-turbo-instruct": "#d62728",  # muted red
    "gpt-4-turbo-2024-04-09": "#ff7f0e",  # muted orange
    "gpt-4-0125-preview": "#ff7f0e",      # muted orange
    "gpt-4-0613": "#ff7f0e",              # muted orange
    "gpt-4o-2024-11-20": "#d4af37",       # dark yellow
    "gpt-4o-2024-08-06": "#d4af37",       # dark yellow
    "gpt-4o-2024-05-13": "#d4af37",       # dark yellow
    "claude-3-5-sonnet-20241022": "#1f77b4",  # blue
    "claude-3-5-sonnet-20240620": "#1f77b4",  # blue
    "claude-3-5-sonnet-latest": "#1f77b4",     # blue
    "claude-3-5-haiku-20241022": "#2ca02c",    # muted green
    "claude-3-5-haiku-latest": "#2ca02c",      # muted green
    "claude-3-7-sonnet-20250219": "#9467bd",    # muted purple
    "claude-3-7-sonnet-latest": "#9467bd"      # muted purple
}

def update_graph(csv_path, output_path, temperature):
    """Update graph with specific order and colors."""
    
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Get Yes counts for each model
    yes_counts = {}
    for model in MODEL_ORDER:
        model_data = df[df['model'] == model]
        if len(model_data) > 0:
            yes_count = model_data.iloc[0]['n_yes']
        else:
            yes_count = 0
        yes_counts[model] = yes_count
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Create bars with specific colors
    bars = plt.bar(range(len(MODEL_ORDER)), 
                   [yes_counts[model] for model in MODEL_ORDER],
                   color=[MODEL_COLORS[model] for model in MODEL_ORDER])
    
    # Customize the plot
    plt.title(f'Historical Models Baby Stroller Test (Temperature {temperature})\nNumber of "Yes" Responses per Model', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Number of "Yes" Responses', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, model) in enumerate(zip(bars, MODEL_ORDER)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Set y-axis limits
    plt.ylim(0, 105)
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Updated graph saved to: {output_path}")

def main():
    """Update both temperature 0.0 and 0.3 graphs."""
    
    # Update temperature 0.0 graph
    temp_0_csv = "historical_models_baby_stroller_results/model_summary.csv"
    temp_0_output = "historical_models_baby_stroller_results/historical_models_baby_stroller_graph_updated.png"
    
    if Path(temp_0_csv).exists():
        update_graph(temp_0_csv, temp_0_output, "0.0")
    else:
        print(f"Warning: {temp_0_csv} not found")
    
    # Update temperature 0.3 graph
    temp_03_csv = "temperature_0.3_tests/temperature_0.3_tests/historical_models_baby_stroller_results_temp_0.3/model_summary.csv"
    temp_03_output = "temperature_0.3_tests/temperature_0.3_tests/historical_models_baby_stroller_results_temp_0.3/historical_models_baby_stroller_graph_temp_0.3_updated.png"
    
    if Path(temp_03_csv).exists():
        update_graph(temp_03_csv, temp_03_output, "0.3")
    else:
        print(f"Warning: {temp_03_csv} not found")

if __name__ == "__main__":
    main()
