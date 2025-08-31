#!/usr/bin/env python3
"""
Chart Generator for LLM Yes/No Experiments
Creates three vertical bar charts for each question
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

from experiment_config import *

# Configure matplotlib and seaborn for professional styling
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generates three vertical bar charts for each question"""
    
    def __init__(self):
        self.historical_models = []
        self.new_models = []
        self._setup_model_lists()
    
    def _setup_model_lists(self):
        """Setup lists of historical and new models"""
        for provider, models in HISTORICAL_MODELS.items():
            self.historical_models.extend(models)
        
        for provider, models in NEW_MODELS.items():
            self.new_models.extend(models)
    
    def generate_question_charts(self, question_data: Dict, results: List[Dict], question_idx: int):
        """Generate all three charts for a specific question"""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            if df.empty:
                logger.warning(f"No results for question {question_idx + 1}")
                return
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'Question {question_idx + 1}: {question_data["question"][:80]}{"..." if len(question_data["question"]) > 80 else ""}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Generate each chart
            self._generate_chart_1_historical_models(ax1, df, question_data)
            self._generate_chart_2_new_models_prompts(ax2, df, question_data)
            self._generate_chart_3_new_models_temperatures(ax3, df, question_data)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save chart
            chart_file = CHARTS_DIR / f"question_{question_idx + 1}_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved charts for question {question_idx + 1} to {chart_file}")
            
        except Exception as e:
            logger.error(f"Error generating charts for question {question_idx + 1}: {e}")
    
    def _generate_chart_1_historical_models(self, ax, df: pd.DataFrame, question_data: Dict):
        """Chart 1: Historical models showing % Yes across 10 trials"""
        try:
            # Define the specific order for historical models
            historical_model_order = [
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-instruct", 
                "gpt-3.5-turbo-0125",
                "gpt-4o-2024-05-13",
                "gpt-4o-2024-08-06",
                "gpt-4o-2024-11-20",
                "gpt-4-0125-preview",
                "gpt-4-0613",
                "gpt-4-turbo-2024-04-09",
                "claude-3-5-haiku-20241022",
                "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-latest",
                "claude-3-7-sonnet-20250219",
                "claude-3-7-sonnet-latest"
            ]
            
            # Filter for historical models and calculate % Yes
            historical_data = []
            
            for model in historical_model_order:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    # Calculate % Yes across all trials for this model
                    total_trials = len(model_df)
                    yes_count = model_df['is_yes'].sum()
                    percent_yes = (yes_count / total_trials * 100) if total_trials > 0 else 0
                    
                    historical_data.append({
                        'model': model,
                        'percent_yes': percent_yes,
                        'total_trials': total_trials
                    })
            
            if not historical_data:
                ax.text(0.5, 0.5, 'No historical model data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Chart 1: Historical Models\n% Yes Across All Trials')
                return
            
            # Create DataFrame and maintain the specified order
            historical_df = pd.DataFrame(historical_data)
            
            # Create bar chart
            bars = ax.bar(range(len(historical_df)), historical_df['percent_yes'], 
                         color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
            
            # Customize chart
            ax.set_title('Chart 1: Historical Models\n% Yes Across All Trials', fontweight='bold', pad=20)
            ax.set_xlabel('Models')
            ax.set_ylabel('% Yes')
            ax.set_ylim(0, 100)
            
            # Set x-axis labels
            ax.set_xticks(range(len(historical_df)))
            ax.set_xticklabels(historical_df['model'], rotation=45, ha='right')
            
            # Add value labels on bars with increased spacing
            for i, (bar, percent, trials) in enumerate(zip(bars, historical_df['percent_yes'], historical_df['total_trials'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                       f'{percent:.1f}%\n({trials} trials)', 
                       ha='center', va='bottom', fontsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Error generating Chart 1: {e}")
            ax.text(0.5, 0.5, f'Error generating chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _generate_chart_2_new_models_prompts(self, ax, df: pd.DataFrame, question_data: Dict):
        """Chart 2: New models with three prompt variations"""
        try:
            # Filter for new models and calculate % Yes for each prompt variation
            prompt_data = []
            
            for model in self.new_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    for prompt_var in PROMPT_VARIATIONS.keys():
                        prompt_df = model_df[model_df['prompt_variation'] == prompt_var]
                        if not prompt_df.empty:
                            # Calculate % Yes across all temperatures for this model+prompt
                            total_trials = len(prompt_df)
                            yes_count = prompt_df['is_yes'].sum()
                            percent_yes = (yes_count / total_trials * 100) if total_trials > 0 else 0
                            
                            prompt_data.append({
                                'model': model,
                                'prompt_variation': prompt_var,
                                'percent_yes': percent_yes,
                                'total_trials': total_trials
                            })
            
            if not prompt_data:
                ax.text(0.5, 0.5, 'No new model data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Chart 2: New Models\n% Yes by Prompt Variation')
                return
            
            # Create DataFrame
            prompt_df = pd.DataFrame(prompt_data)
            
            # Create grouped bar chart
            models = prompt_df['model'].unique()
            prompt_vars = list(PROMPT_VARIATIONS.keys())
            
            x = np.arange(len(models))
            width = 0.25
            
            # Plot bars for each prompt variation
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, prompt_var in enumerate(prompt_vars):
                data = prompt_df[prompt_df['prompt_variation'] == prompt_var]
                if not data.empty:
                    # Align data with models
                    aligned_data = []
                    for model in models:
                        model_data = data[data['model'] == model]
                        if not model_data.empty:
                            aligned_data.append(model_data.iloc[0]['percent_yes'])
                        else:
                            aligned_data.append(0)
                    
                    bars = ax.bar(x + i * width, aligned_data, width, 
                                 label=prompt_var.replace('_', ' ').title(), 
                                 color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add value labels with increased spacing
                    for j, (bar, percent) in enumerate(zip(bars, aligned_data)):
                        if percent > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                                   f'{percent:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Customize chart
            ax.set_title('Chart 2: New Models\n% Yes by Prompt Variation', fontweight='bold', pad=20)
            ax.set_xlabel('Models')
            ax.set_ylabel('% Yes')
            ax.set_ylim(0, 100)
            
            # Set x-axis labels
            ax.set_xticks(x + width)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add legend
            ax.legend(title='Prompt Variation', loc='upper right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Error generating Chart 2: {e}")
            ax.text(0.5, 0.5, f'Error generating chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _generate_chart_3_new_models_temperatures(self, ax, df: pd.DataFrame, question_data: Dict):
        """Chart 3: New models with three temperature settings"""
        try:
            # Filter for new models and calculate % Yes for each temperature
            temp_data = []
            
            for model in self.new_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    for temp in TEMPERATURES:
                        temp_df = model_df[model_df['temperature'] == temp]
                        if not temp_df.empty:
                            # Calculate % Yes across all prompt variations for this model+temp
                            total_trials = len(temp_df)
                            yes_count = temp_df['is_yes'].sum()
                            percent_yes = (yes_count / total_trials * 100) if total_trials > 0 else 0
                            
                            temp_data.append({
                                'model': model,
                                'temperature': temp,
                                'percent_yes': percent_yes,
                                'total_trials': total_trials
                            })
            
            if not temp_data:
                ax.text(0.5, 0.5, 'No new model data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Chart 3: New Models\n% Yes by Temperature')
                return
            
            # Create DataFrame
            temp_df = pd.DataFrame(temp_data)
            
            # Create grouped bar chart
            models = temp_df['model'].unique()
            temps = TEMPERATURES
            
            x = np.arange(len(models))
            width = 0.25
            
            # Plot bars for each temperature
            colors = ['#FFD93D', '#FF6B6B', '#6BCF7F']
            for i, temp in enumerate(temps):
                data = temp_df[temp_df['temperature'] == temp]
                if not data.empty:
                    # Align data with models
                    aligned_data = []
                    for model in models:
                        model_data = data[data['model'] == model]
                        if not model_data.empty:
                            aligned_data.append(model_data.iloc[0]['percent_yes'])
                        else:
                            aligned_data.append(0)
                    
                    bars = ax.bar(x + i * width, aligned_data, width, 
                                 label=f'Temp {temp}', 
                                 color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add value labels with increased spacing
                    for j, (bar, percent) in enumerate(zip(bars, aligned_data)):
                        if percent > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                                   f'{percent:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Customize chart
            ax.set_title('Chart 3: New Models\n% Yes by Temperature', fontweight='bold', pad=20)
            ax.set_xlabel('Models')
            ax.set_ylabel('% Yes')
            ax.set_ylim(0, 100)
            
            # Set x-axis labels
            ax.set_xticks(x + width)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add legend
            ax.legend(title='Temperature', loc='upper right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Error generating Chart 3: {e}")
            ax.text(0.5, 0.5, f'Error generating chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _generate_summary_chart(self, df: pd.DataFrame, question_data: Dict):
        """Generate a summary chart showing all models together"""
        try:
            # Calculate % Yes for all models
            summary_data = []
            
            # Add historical models
            for model in self.historical_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    total_trials = len(model_df)
                    yes_count = model_df['is_yes'].sum()
                    percent_yes = (yes_count / total_trials * 100) if total_trials > 0 else 0
                    
                    summary_data.append({
                        'model': model,
                        'category': 'Historical',
                        'percent_yes': percent_yes,
                        'total_trials': total_trials
                    })
            
            # Add new models
            for model in self.new_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    total_trials = len(model_df)
                    yes_count = model_df['is_yes'].sum()
                    percent_yes = (yes_count / total_trials * 100) if total_trials > 0 else 0
                    
                    summary_data.append({
                        'model': model,
                        'category': 'New',
                        'percent_yes': percent_yes,
                        'total_trials': total_trials
                    })
            
            if not summary_data:
                logger.warning("No summary data available")
                return
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Create bar chart with different colors for historical vs new
            colors = ['skyblue' if cat == 'Historical' else 'lightcoral' for cat in summary_df['category']]
            
            bars = ax.bar(range(len(summary_df)), summary_df['percent_yes'], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize chart
            ax.set_title(f'Summary: All Models - Question {question_data.get("question_index", "Unknown")}\n% Yes Across All Trials', 
                        fontweight='bold', pad=20)
            ax.set_xlabel('Models')
            ax.set_ylabel('% Yes')
            ax.set_ylim(0, 100)
            
            # Set x-axis labels
            ax.set_xticks(range(len(summary_df)))
            ax.set_xticklabels(summary_df['model'], rotation=45, ha='right')
            
            # Add value labels
            for i, (bar, percent, trials) in enumerate(zip(bars, summary_df['percent_yes'], summary_df['total_trials'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                       f'{percent:.1f}%\n({trials} trials)', 
                       ha='center', va='bottom', fontsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='skyblue', alpha=0.8, label='Historical Models'),
                Patch(facecolor='lightcoral', alpha=0.8, label='New Models')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            summary_file = CHARTS_DIR / f"question_{question_data.get('question_index', 'unknown')}_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved summary chart to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary chart: {e}")


if __name__ == "__main__":
    # Test the chart generator
    generator = ChartGenerator()
    print("Historical models:", generator.historical_models)
    print("New models:", generator.new_models)
