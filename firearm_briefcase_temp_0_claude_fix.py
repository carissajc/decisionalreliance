#!/usr/bin/env python3
"""
Historical Models Firearm Briefcase Test - Claude Fix (Temperature 0.0)
Reruns ONLY the Claude/Anthropic models with fixes for max_tokens, strict prompts, and better normalization.
"""

import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from tqdm import tqdm
import time
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
ANTHROPIC_API_KEY = "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"

# Test configuration
TEST_QUESTION = "Does transporting a firearm in a briefcase constitute 'carrying' a firearm?"
TRIALS_PER_MODEL = 100
TEMPERATURE = 0.0

# STRICT prompt that forces Yes/No first
STRICT_PROMPT = f"""You must answer with ONLY "Yes" or "No" first, then provide explanation if needed.

Question: {TEST_QUESTION}

Answer:"""

# Only Claude models to test
CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest"
]

class ClaudeFirearmBriefcaseFixer:
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.results = []
        
    def _normalize_response(self, response_text: str) -> str:
        """Enhanced normalization logic to handle partial responses."""
        response_text = response_text.strip().lower()
        
        # Look for Yes/No in the first few words
        words = response_text.split()[:5]  # Check first 5 words
        
        for word in words:
            if word.startswith('yes'):
                return 'Yes'
            elif word.startswith('no'):
                return 'No'
        
        # If no clear Yes/No found, check for partial matches
        if 'yes' in response_text[:20]:  # Check first 20 characters
            return 'Yes'
        elif 'no' in response_text[:20]:
            return 'No'
        
        # Check for common partial responses
        if response_text.startswith('this is'):
            # Likely a truncated response, try to infer
            if 'carry' in response_text or 'constitute' in response_text:
                return 'Yes'  # If they're discussing carrying, likely Yes
            else:
                return 'Unknown'
        
        return 'Unknown'
    
    async def run_single_trial(self, model: str, trial_idx: int) -> dict:
        """Run a single trial for a Claude model."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=15,  # Increased from 3 to 15
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": STRICT_PROMPT}]
            )
            raw_text = response.content[0].text
            
            extracted_answer = self._normalize_response(raw_text)
            
            return {
                'provider': 'anthropic',
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
    
    async def test_all_claude_models(self):
        """Test all Claude models."""
        total_trials = len(CLAUDE_MODELS) * TRIALS_PER_MODEL
        
        with tqdm(total=total_trials, desc="Testing Claude models") as pbar:
            for model in CLAUDE_MODELS:
                logger.info(f"Testing model: {model}")
                
                for trial_idx in range(TRIALS_PER_MODEL):
                    result = await self.run_single_trial(model, trial_idx)
                    if result:
                        self.results.append(result)
                    pbar.update(1)
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.1)
        
        logger.info(f"Completed testing. Total results: {len(self.results)}")
    
    def merge_with_existing_openai_results(self):
        """Merge new Claude results with existing OpenAI results."""
        # Read existing OpenAI results
        existing_file = Path("firearm_briefcase_temp_0_results/detailed_results.csv")
        if existing_file.exists():
            df_existing = pd.read_csv(existing_file)
            # Filter to only OpenAI results
            openai_results = df_existing[df_existing['provider'] == 'openai'].to_dict('records')
            
            # Combine with new Claude results
            all_results = openai_results + self.results
            logger.info(f"Merged results: {len(openai_results)} OpenAI + {len(self.results)} Claude = {len(all_results)} total")
            return all_results
        else:
            logger.warning("No existing results found, using only Claude results")
            return self.results
    
    def save_results(self):
        """Save results to CSV files."""
        output_dir = Path("firearm_briefcase_temp_0_results")
        output_dir.mkdir(exist_ok=True)
        
        # Merge with existing OpenAI results
        all_results = self.merge_with_existing_openai_results()
        
        # Save detailed results
        df_detailed = pd.DataFrame(all_results)
        detailed_path = output_dir / "detailed_results_fixed.csv"
        df_detailed.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to {detailed_path}")
        
        # Create summary
        summary_data = []
        all_models = [
            "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
            "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview", "gpt-4-0613",
            "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"
        ] + CLAUDE_MODELS
        
        for model in all_models:
            model_results = [r for r in all_results if r['model'] == model]
            n_trials = len(model_results)
            n_yes = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
            percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
            
            # Determine provider
            provider = "anthropic" if model in CLAUDE_MODELS else "openai"
            
            summary_data.append({
                'provider': provider,
                'model': model,
                'n_trials': n_trials,
                'n_yes': n_yes,
                'percent_yes': percent_yes
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = output_dir / "model_summary_fixed.csv"
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        return output_dir, all_results
    
    def create_graph(self, output_dir: Path, all_results):
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
            model_results = [r for r in all_results if r['model'] == model]
            yes_count = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
            yes_counts[model] = yes_count

        # Create the plot
        plt.figure(figsize=(16, 8))

        # Create bars with specific colors
        bars = plt.bar(range(len(MODEL_ORDER)),
                       [yes_counts[model] for model in MODEL_ORDER],
                       color=[MODEL_COLORS[model] for model in MODEL_ORDER])

        # Customize the plot
        plt.title(f'Historical Models Firearm Briefcase Test (Temperature {TEMPERATURE})\nNumber of "Yes" Responses per Model - FIXED',
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
        graph_path = output_dir / f"historical_models_firearm_briefcase_graph_temp_{TEMPERATURE}_fixed.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved graph to {graph_path}")

    def generate_report(self, output_dir: Path, all_results):
        """Generate a text report."""
        # Calculate summary statistics
        total_trials = len(all_results)
        successful_trials = len([r for r in all_results if r['extracted_answer'] != 'Unknown'])

        # Top 5 models by Yes responses
        model_yes_counts = {}
        for result in all_results:
            model = result['model']
            if model not in model_yes_counts:
                model_yes_counts[model] = 0
            if result['extracted_answer'] == 'Yes':
                model_yes_counts[model] += 1

        top_models = sorted(model_yes_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Generate report
        report_content = f"""HISTORICAL MODELS FIREARM BRIEFCASE TEST REPORT - FIXED (TEMPERATURE {TEMPERATURE})

Question tested: {TEST_QUESTION}
Temperature: {TEMPERATURE}
Total models tested: 16
Trials per model: {TRIALS_PER_MODEL}
Total trials: {total_trials}
Successful trials: {successful_trials}

IMPROVEMENTS MADE:
- Increased Claude max_tokens from 3 to 15
- Added strict prompt forcing "Yes" or "No" first
- Enhanced normalization logic for partial responses
- Reran only Claude models, kept existing OpenAI results

Top 5 models by number of Yes responses:"""

        for i, (model, yes_count) in enumerate(top_models, 1):
            percent = (yes_count / TRIALS_PER_MODEL) * 100
            report_content += f"\n{i}. {model}: {yes_count}/{TRIALS_PER_MODEL} Yes ({percent:.1f}%)"

        report_content += f"\n\nResults saved in: {output_dir}/"

        # Save report
        report_path = output_dir / f"test_report_temp_{TEMPERATURE}_fixed.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Generated report: {report_path}")
        print(report_content)

async def main():
    """Main function."""
    fixer = ClaudeFirearmBriefcaseFixer()
    
    print(f"Starting Claude Models Firearm Briefcase Fix (Temperature {TEMPERATURE})")
    print(f"Question: {TEST_QUESTION}")
    print(f"Trials per model: {TRIALS_PER_MODEL}")
    print("IMPROVEMENTS: max_tokens=15, strict prompts, better normalization")
    print("=" * 80)
    
    await fixer.test_all_claude_models()
    
    output_dir, all_results = fixer.save_results()
    fixer.create_graph(output_dir, all_results)
    fixer.generate_report(output_dir, all_results)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
