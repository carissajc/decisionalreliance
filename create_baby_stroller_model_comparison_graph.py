#!/usr/bin/env python3
"""
Create Baby Stroller Model Comparison Graph
Shows each model on x-axis with their percentage "Yes" responses for all prompts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_model_comparison_graph():
    # Read the detailed results
    results_file = Path("new_models_baby_stroller_prompt_test_results_temp_0.7/detailed_results.csv")
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    
    # Get unique models and prompts
    models = df['model'].unique()
    prompts = df['prompt_name'].unique()
    
    # Calculate percentage "Yes" for each model-prompt combination
    model_prompt_stats = []
    for model in models:
        for prompt in prompts:
            model_prompt_data = df[(df['model'] == model) & (df['prompt_name'] == prompt)]
            if len(model_prompt_data) > 0:
                n_yes = len(model_prompt_data[model_prompt_data['extracted_answer'] == 'Yes'])
                total = len(model_prompt_data)
                percent_yes = (n_yes / total * 100) if total > 0 else 0
                model_prompt_stats.append({
                    'model': model,
                    'prompt': prompt,
                    'percent_yes': percent_yes,
                    'n_yes': n_yes,
                    'total': total
                })
    
    # Create DataFrame for easier manipulation
    stats_df = pd.DataFrame(model_prompt_stats)
    
    # Define muted colors for models
    model_colors = {
        'gpt-4.1-2025-04-14': '#d62728',      # muted red
        'claude-sonnet-4-20250514': '#1f77b4',  # muted blue
        'grok-4-0709': '#2ca02c',              # muted green
        'gemini-2.0-flash': '#ff7f0e',         # muted orange
        'Llama-4-Maverick-17B-128E-Instruct-FP8': '#9467bd'  # muted purple
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up the data for plotting
    x_positions = np.arange(len(models))
    bar_width = 0.15  # Width of each bar group
    
    # Plot bars for each prompt
    for i, prompt in enumerate(prompts):
        prompt_data = stats_df[stats_df['prompt'] == prompt]
        
        # Get values for each model in the correct order
        values = []
        for model in models:
            model_data = prompt_data[prompt_data['model'] == model]
            if len(model_data) > 0:
                values.append(model_data.iloc[0]['percent_yes'])
            else:
                values.append(0)
        
        # Calculate x positions for this prompt
        x_pos = x_positions + (i - len(prompts)/2 + 0.5) * bar_width
        
        # Create bars
        bars = ax.bar(x_pos, values, bar_width, 
                     label=prompt, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage "Yes" Responses', fontsize=14, fontweight='bold')
    ax.set_title('Baby Stroller Prompt Variations Test - Model Comparison\nPercentage "Yes" by Model and Prompt (Temperature 0.7)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("new_models_baby_stroller_prompt_test_results_temp_0.7")
    graph_path = output_dir / "baby_stroller_model_comparison_graph.png"
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison graph saved to: {graph_path}")
    
    # Also create a summary table
    summary_table = stats_df.pivot(index='model', columns='prompt', values='percent_yes')
    summary_table = summary_table.round(1)
    
    # Save summary table
    summary_path = output_dir / "model_prompt_summary_table.csv"
    summary_table.to_csv(summary_path)
    print(f"Summary table saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL")
    print("="*80)
    
    for model in models:
        model_data = stats_df[stats_df['model'] == model]
        avg_percent = model_data['percent_yes'].mean()
        min_percent = model_data['percent_yes'].min()
        max_percent = model_data['percent_yes'].max()
        
        print(f"\n{model}:")
        print(f"  Average: {avg_percent:.1f}%")
        print(f"  Range: {min_percent:.1f}% - {max_percent:.1f}%")
        
        # Show top 3 prompts for this model
        top_prompts = model_data.nlargest(3, 'percent_yes')
        print("  Top 3 prompts:")
        for _, row in top_prompts.iterrows():
            print(f"    {row['prompt']}: {row['percent_yes']:.1f}%")

if __name__ == "__main__":
    create_model_comparison_graph()
