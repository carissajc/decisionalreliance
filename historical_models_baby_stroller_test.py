#!/usr/bin/env python3
"""
Historical Models Baby Stroller Test
Tests all historical models with the baby stroller question, 100 trials per model at temperature 0.0.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"

# Test question
TEST_QUESTION = "Is a 'baby stroller' a 'vehicle'?"

# Number of trials per model
TRIALS_PER_MODEL = 100

# Temperature (0.0 for deterministic responses)
TEMPERATURE = 0.0

# Historical models to test
HISTORICAL_MODELS = {
    "openai": [
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106", 
        "gpt-3.5-turbo-instruct",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest"
    ]
}

class HistoricalModelsBabyStrollerTester:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.results = []
        
    def _normalize_response(self, text: str) -> Tuple[str, str, Optional[int]]:
        """Parse response to extract Yes/No and calculate is_yes"""
        raw_text = text.strip() if text else ""
        
        # Extract Yes/No from the beginning of the response
        if raw_text.lower().startswith("yes"):
            extracted_answer = "Yes"
            is_yes = 1
        elif raw_text.lower().startswith("no"):
            extracted_answer = "No"
            is_yes = 0
        else:
            extracted_answer = raw_text
            is_yes = None
            
        return raw_text, extracted_answer, is_yes
    
    async def run_single_trial(self, provider: str, model: str, trial_idx: int) -> Optional[Dict]:
        """Run a single trial for a model"""
        try:
            # System message to enforce Yes/No format
            system_message = "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else."
            
            if provider == "openai":
                # Handle different model types
                if model == "gpt-3.5-turbo-instruct":
                    response = self.openai_client.completions.create(
                        model=model,
                        prompt=f"{system_message}\n\n{TEST_QUESTION}",
                        max_tokens=3,
                        temperature=TEMPERATURE
                    )
                    raw_response = response.choices[0].text
                else:
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": TEST_QUESTION}
                        ],
                        max_tokens=3,
                        temperature=TEMPERATURE
                    )
                    raw_response = response.choices[0].message.content
                    
            elif provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=3,
                    temperature=TEMPERATURE,
                    system=system_message,
                    messages=[{"role": "user", "content": TEST_QUESTION}]
                )
                raw_response = response.content[0].text if response.content else ""
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            raw_text, extracted_answer, is_yes = self._normalize_response(raw_response)
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "model": model,
                "question": TEST_QUESTION,
                "raw_response": raw_response,
                "extracted_answer": extracted_answer,
                "is_yes": is_yes,
                "trial_idx": trial_idx,
                "temperature": TEMPERATURE
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Trial failed for {provider}/{model} trial {trial_idx}: {e}")
            return None
    
    async def test_all_historical_models(self):
        """Test all historical models with multiple trials"""
        logger.info(f"Starting historical models testing with {TRIALS_PER_MODEL} trials per model")
        logger.info(f"Testing question: {TEST_QUESTION}")
        logger.info(f"Temperature: {TEMPERATURE}")
        
        # Count total models
        total_models = sum(len(models) for models in HISTORICAL_MODELS.values())
        total_trials = total_models * TRIALS_PER_MODEL
        
        logger.info(f"Total models to test: {total_models}")
        logger.info(f"Total trials: {total_trials}")
        
        with tqdm(total=total_trials, desc="Testing historical models") as pbar:
            for provider, models in HISTORICAL_MODELS.items():
                logger.info(f"Testing {provider} models...")
                
                for model in models:
                    logger.info(f"Testing model: {model}")
                    
                    for trial_idx in range(TRIALS_PER_MODEL):
                        result = await self.run_single_trial(provider, model, trial_idx)
                        
                        if result:
                            self.results.append(result)
                        
                        pbar.update(1)
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
        
        logger.info(f"Completed testing. Total results: {len(self.results)}")
    
    def save_results(self):
        """Save all results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create output directory
        output_dir = Path("historical_models_baby_stroller_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        results_file = output_dir / "detailed_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved detailed results to {results_file}")
        
        # Create summary by model
        summary = df.groupby(['provider', 'model']).agg({
            'is_yes': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        
        summary.columns = ['provider', 'model', 'n_trials', 'n_yes', 'percent_yes', 'std_dev']
        summary['percent_yes'] = (summary['percent_yes'] * 100).round(1)
        summary['std_dev'] = summary['std_dev'].round(3)
        
        summary_file = output_dir / "model_summary.csv"
        summary.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        return summary
    
    def create_graph(self, summary_df: pd.DataFrame):
        """Create a graph showing number of Yes responses for each model"""
        try:
            # Create the plot
            plt.figure(figsize=(16, 10))
            
            # Sort by number of Yes responses for better visualization
            summary_df = summary_df.sort_values('n_yes', ascending=False)
            
            # Create bars
            bars = plt.bar(range(len(summary_df)), summary_df['n_yes'], 
                          color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
            
            # Customize the plot
            plt.title(f'Historical Models Response to: "{TEST_QUESTION}"\n'
                     f'{TRIALS_PER_MODEL} trials per model | Temperature: {TEMPERATURE}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Model', fontweight='bold', fontsize=12)
            plt.ylabel('Number of "Yes" Responses', fontweight='bold', fontsize=12)
            
            # Set x-axis labels
            plt.xticks(range(len(summary_df)), 
                      [f"{row['provider']}\n{row['model'].replace('gpt-', '').replace('claude-', '')}" 
                       for _, row in summary_df.iterrows()], 
                      rotation=45, ha='right', fontsize=10)
            
            # Add value labels on bars
            for i, (bar, n_yes, total_trials) in enumerate(zip(bars, summary_df['n_yes'], summary_df['n_trials'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{n_yes}/{total_trials}\n({n_yes/total_trials*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=9)
            
            # Add grid
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            output_dir = Path("historical_models_baby_stroller_results")
            plot_file = output_dir / "historical_models_baby_stroller_graph.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved graph to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
    
    def generate_report(self, summary_df: pd.DataFrame):
        """Generate a text report of the results"""
        try:
            output_dir = Path("historical_models_baby_stroller_results")
            report_file = output_dir / "test_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("HISTORICAL MODELS BABY STROLLER TEST REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Test Question: {TEST_QUESTION}\n")
                f.write(f"Trials per model: {TRIALS_PER_MODEL}\n")
                f.write(f"Temperature: {TEMPERATURE}\n")
                f.write(f"Total models tested: {len(summary_df)}\n")
                f.write(f"Total trials: {len(self.results)}\n")
                f.write(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("RESULTS SUMMARY:\n")
                f.write("-" * 20 + "\n")
                
                # Sort by number of Yes responses
                sorted_summary = summary_df.sort_values('n_yes', ascending=False)
                
                for _, row in sorted_summary.iterrows():
                    f.write(f"{row['provider']}/{row['model']}: {row['n_yes']}/{row['n_trials']} Yes ({row['percent_yes']:.1f}%)\n")
                
                f.write(f"\nSTATISTICS:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Highest number of Yes: {sorted_summary['n_yes'].max()}\n")
                f.write(f"Lowest number of Yes: {sorted_summary['n_yes'].min()}\n")
                f.write(f"Mean number of Yes: {sorted_summary['n_yes'].mean():.1f}\n")
                f.write(f"Standard Deviation: {sorted_summary['n_yes'].std():.1f}\n")
                
                # Find most and least Yes models
                most_yes = sorted_summary.iloc[0]
                least_yes = sorted_summary.iloc[-1]
                
                f.write(f"\nMOST YES RESPONSES:\n")
                f.write(f"  {most_yes['provider']}/{most_yes['model']}: {most_yes['n_yes']}/{most_yes['n_trials']} ({most_yes['percent_yes']:.1f}%)\n")
                
                f.write(f"\nLEAST YES RESPONSES:\n")
                f.write(f"  {least_yes['provider']}/{least_yes['model']}: {least_yes['n_yes']}/{least_yes['n_trials']} ({least_yes['percent_yes']:.1f}%)\n")
                
                # Provider comparison
                f.write(f"\nPROVIDER COMPARISON:\n")
                f.write("-" * 25 + "\n")
                provider_summary = summary_df.groupby('provider')['n_yes'].agg(['sum', 'mean', 'count'])
                for provider, row in provider_summary.iterrows():
                    f.write(f"{provider}: {row['sum']} total Yes, {row['mean']:.1f} avg Yes per model ({row['count']} models)\n")
            
            logger.info(f"Generated report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


async def main():
    """Main function to run the historical models testing"""
    logger.info("Starting historical models baby stroller test...")
    
    # Create tester
    tester = HistoricalModelsBabyStrollerTester()
    
    try:
        # Run all tests
        await tester.test_all_historical_models()
        
        # Save results
        summary = tester.save_results()
        
        if summary is not None:
            # Create graph
            tester.create_graph(summary)
            
            # Generate report
            tester.generate_report(summary)
            
            logger.info("Testing completed successfully!")
            logger.info(f"Results saved in: historical_models_baby_stroller_results/")
            
            # Print summary to console
            print("\n" + "="*80)
            print("HISTORICAL MODELS BABY STROLLER TEST COMPLETED")
            print("="*80)
            print(f"Question tested: {TEST_QUESTION}")
            print(f"Total models tested: {len(summary)}")
            print(f"Trials per model: {TRIALS_PER_MODEL}")
            print(f"Total trials: {len(tester.results)}")
            print(f"\nTop 5 models by number of Yes responses:")
            
            sorted_summary = summary.sort_values('n_yes', ascending=False)
            for i, (_, row) in enumerate(sorted_summary.head().iterrows()):
                print(f"{i+1}. {row['provider']}/{row['model']}: {row['n_yes']}/{row['n_trials']} Yes ({row['percent_yes']:.1f}%)")
            
            print(f"\nResults saved in: historical_models_baby_stroller_results/")
            
        else:
            logger.error("No results to process")
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
