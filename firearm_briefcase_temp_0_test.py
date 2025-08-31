#!/usr/bin/env python3
"""
Historical Models Firearm Briefcase Test (Temperature 0.0)
Tests all historical models with the firearm briefcase question, 100 trials per model at temperature 0.0.
"""

import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from tqdm import tqdm
import time
from openai import OpenAI
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"

# Test configuration
TEST_QUESTION = "Does transporting a firearm in a briefcase constitute 'carrying' a firearm?"
TRIALS_PER_MODEL = 100
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

class HistoricalModelsFirearmBriefcaseTester:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.results = []
        
    def _normalize_response(self, response_text: str) -> str:
        """Extract Yes/No from response."""
        response_text = response_text.strip().lower()
        if response_text.startswith('yes'):
            return 'Yes'
        elif response_text.startswith('no'):
            return 'No'
        else:
            return 'Unknown'
    
    async def run_single_trial(self, provider: str, model: str, trial_idx: int) -> dict:
        """Run a single trial for a model."""
        try:
            if provider == "openai":
                if model == "gpt-3.5-turbo-instruct":
                    # Use completions endpoint for gpt-3.5-turbo-instruct
                    response = self.openai_client.completions.create(
                        model=model,
                        prompt=TEST_QUESTION,
                        temperature=TEMPERATURE,
                        max_tokens=3
                    )
                    raw_text = response.choices[0].text
                else:
                    # Use chat completions endpoint for other models
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": TEST_QUESTION}],
                        temperature=TEMPERATURE,
                        max_tokens=3
                    )
                    raw_text = response.choices[0].message.content
            elif provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=3,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": TEST_QUESTION}]
                )
                raw_text = response.content[0].text
            else:
                return None
            
            extracted_answer = self._normalize_response(raw_text)
            
            return {
                'provider': provider,
                'model': model,
                'trial': trial_idx,
                'question': TEST_QUESTION,
                'raw_text': raw_text,
                'extracted_answer': extracted_answer,
                'temperature': TEMPERATURE
            }
            
        except Exception as e:
            logger.error(f"Error in trial {trial_idx} for {model}: {e}")
            return None
    
    async def test_all_historical_models(self):
        """Test all historical models."""
        total_trials = sum(len(models) for models in HISTORICAL_MODELS.values()) * TRIALS_PER_MODEL
        
        with tqdm(total=total_trials, desc="Testing historical models") as pbar:
            for provider, models in HISTORICAL_MODELS.items():
                for model in models:
                    logger.info(f"Testing model: {model}")
                    
                    for trial_idx in range(TRIALS_PER_MODEL):
                        result = await self.run_single_trial(provider, model, trial_idx)
                        if result:
                            self.results.append(result)
                        pbar.update(1)
                        
                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)
        
        logger.info(f"Completed testing. Total results: {len(self.results)}")
    
    def save_results(self):
        """Save results to CSV files."""
        output_dir = Path("firearm_briefcase_temp_0_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df_detailed = pd.DataFrame(self.results)
        detailed_path = output_dir / "detailed_results.csv"
        df_detailed.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to {detailed_path}")
        
        # Create summary
        summary_data = []
        for provider, models in HISTORICAL_MODELS.items():
            for model in models:
                model_results = [r for r in self.results if r['model'] == model]
                n_trials = len(model_results)
                n_yes = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
                percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
                
                summary_data.append({
                    'provider': provider,
                    'model': model,
                    'n_trials': n_trials,
                    'n_yes': n_yes,
                    'percent_yes': percent_yes
                })
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = output_dir / "model_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        return output_dir
    
    def create_graph(self, output_dir: Path):
        """Create bar chart of results."""
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
        
        # Get Yes counts for each model
        yes_counts = {}
        for model in MODEL_ORDER:
            model_results = [r for r in self.results if r['model'] == model]
            yes_count = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
            yes_counts[model] = yes_count
        
        # Create the plot
        plt.figure(figsize=(16, 8))
        
        # Create bars with specific colors
        bars = plt.bar(range(len(MODEL_ORDER)), 
                       [yes_counts[model] for model in MODEL_ORDER],
                       color=[MODEL_COLORS[model] for model in MODEL_ORDER])
        
        # Customize the plot
        plt.title(f'Historical Models Firearm Briefcase Test (Temperature {TEMPERATURE})\nNumber of "Yes" Responses per Model', 
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
        graph_path = output_dir / f"historical_models_firearm_briefcase_graph_temp_{TEMPERATURE}.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved graph to {graph_path}")
    
    def generate_report(self, output_dir: Path):
        """Generate a text report."""
        # Calculate summary statistics
        total_trials = len(self.results)
        successful_trials = len([r for r in self.results if r['extracted_answer'] != 'Unknown'])
        
        # Top 5 models by Yes responses
        model_yes_counts = {}
        for provider, models in HISTORICAL_MODELS.items():
            for model in models:
                model_results = [r for r in self.results if r['model'] == model]
                yes_count = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
                model_yes_counts[model] = yes_count
        
        top_models = sorted(model_yes_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report
        report_content = f"""HISTORICAL MODELS FIREARM BRIEFCASE TEST REPORT (TEMPERATURE {TEMPERATURE})

Question tested: {TEST_QUESTION}
Temperature: {TEMPERATURE}
Total models tested: {sum(len(models) for models in HISTORICAL_MODELS.values())}
Trials per model: {TRIALS_PER_MODEL}
Total trials: {total_trials}
Successful trials: {successful_trials}

Top 5 models by number of Yes responses:"""
        
        for i, (model, yes_count) in enumerate(top_models, 1):
            percent = (yes_count / TRIALS_PER_MODEL) * 100
            report_content += f"\n{i}. {model}: {yes_count}/{TRIALS_PER_MODEL} Yes ({percent:.1f}%)"
        
        report_content += f"\n\nResults saved in: {output_dir}/"
        
        # Save report
        report_path = output_dir / f"test_report_temp_{TEMPERATURE}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated report: {report_path}")
        print(report_content)

async def main():
    """Main function."""
    tester = HistoricalModelsFirearmBriefcaseTester()
    
    print(f"Starting Historical Models Firearm Briefcase Test (Temperature {TEMPERATURE})")
    print(f"Question: {TEST_QUESTION}")
    print(f"Trials per model: {TRIALS_PER_MODEL}")
    print("=" * 80)
    
    await tester.test_all_historical_models()
    
    output_dir = tester.save_results()
    tester.create_graph(output_dir)
    tester.generate_report(output_dir)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
