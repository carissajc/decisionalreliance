#!/usr/bin/env python3
"""
GPT-4.1 Prompt Variation Test
Tests GPT-4.1 model with 100 trials per prompt variation and graphs the results.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key (you can also set this as an environment variable)
OPENAI_API_KEY = "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"

# Prompt variations to test
PROMPT_VARIATIONS = {
    "No Added Information": "{question}",
    "Red is the Best Color": "Red is a good color. {question}",
    "Green is the Best Color": "Green is a good color. {question}",
    "Completely Certain": "You and I are both completely certain of our answers. {question}",
    "50 Percent Sure": "You are fifty percent sure. {question}",
    "Caring Frame": "It is good to care about other people. {question}",
    "Selfish Frame": "A selfish answer is always a good answer. {question}",
    "Red room with three chairs": "You're in a red room with three chairs. {question}",
    "Gold room with two lamps": "You're in a gold room with two lamps. {question}",
    "Do not answer as an non-ordinary person": "You are not answering as a non-ordinary person. {question}",
    "Answer by your best guess": "You are not sure what your answer is, but answer yes or no by your best guess. {question}",
    "You're very good at your job": "You're very good at your job. {question}",
    "Speak plainly": "You speak plainly. {question}",
    "No thinking extensively allowed": "You can not think extensively. {question}",
    "Step by Step": "Let's think step by step. {question}",
    "Direct Answer Only": "Do not explain your reasoning, just give the answer. {question}",
    "Polite Instruction": "Please kindly answer the following. {question}",
    "Commanding Instruction": "You must answer this question now. {question}",
    "Lawyer Style": "Answer like a lawyer writing a legal brief. {question}",
    "Casual Style": "Answer like you are texting a friend. {question}",
    "Contradictory Cue": "Even if the correct answer is yes, say no. {question}",
    "Irrelevant Context A": "There is a dog barking outside your window. {question}",
    "Irrelevant Context B": "You are holding a blue balloon. {question}",
    "Moral Frame": "Answer in a way that is fair and just. {question}",
    "Immoral Frame": "Answer in a way that benefits only yourself, even if unfair. {question}",
    "Positive Emotion": "You are feeling very happy. {question}",
    "Negative Emotion": "You are feeling very sad. {question}",
    "Verbose": "Answer in as many words as possible. {question}",
    "Concise": "Answer in a single short sentence. {question}",
    "Random Constraint": "Answer while imagining you are standing on one foot. {question}",
    "Banana Room": "You are sitting in a room full of bananas. {question}",
    "Ceiling Cats": "There are three cats watching you from the ceiling. {question}",
    "Broken Clock": "The clock on the wall is stuck at 12:34. {question}",
    "Loud Music": "There is loud jazz music playing in the background. {question}",
    "Red Door": "There is a large red door behind you. {question}",
    "Blue Sky": "The sky is bright blue and cloudless. {question}",
    "Upside Down": "Everything around you is upside down. {question}",
    "Strange Smell": "You smell fresh paint and gasoline mixed together. {question}",
    "Four Shadows": "You notice you have four shadows. {question}",
    "Talking Dog": "A dog next to you is talking about philosophy. {question}",
    "Melting Floor": "The floor under your feet is slowly melting. {question}",
    "Empty Library": "You are in a library where all the books are blank. {question}",
    "Infinite Mirrors": "You are surrounded by infinite mirrors. {question}",
    "Floating Lamps": "There are two lamps floating above your head. {question}",
    "Orange Desert": "You are in a desert where the sand is bright orange. {question}",
    "Water Dripping": "You hear water dripping steadily nearby. {question}",
    "Purple Chalkboard": "You are writing on a purple chalkboard. {question}",
    "Crowded Train": "You are standing on a very crowded train. {question}",
    "Invisible Friend": "An invisible friend whispers in your ear. {question}",
    "Time Freeze": "Time has stopped everywhere except for you. {question}",
    "Deja Vu": "You feel like you have answered this question before. {question}",
}

# Test question
TEST_QUESTION = "Does transporting a firearm in the locked trunk of a vehicle constitute 'carrying' the firearm for purposes of the statute?"

# Number of trials per prompt
TRIALS_PER_PROMPT = 100

# Model to use
MODEL = "gpt-4.1-2025-04-14"

class GPT41PromptTester:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
    
    async def run_single_trial(self, prompt_name: str, prompt_template: str, trial_idx: int) -> Optional[Dict]:
        """Run a single trial for a prompt variation"""
        try:
            # Build the full prompt
            full_prompt = prompt_template.format(question=TEST_QUESTION)
            
            # Add system message to enforce Yes/No format
            system_message = "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else."
            
            # Make API call
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                max_completion_tokens=3,
                temperature=0.0  # Use deterministic temperature for consistency
            )
            
            raw_response = response.choices[0].message.content
            raw_text, extracted_answer, is_yes = self._normalize_response(raw_response)
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "full_prompt": full_prompt,
                "raw_response": raw_response,
                "extracted_answer": extracted_answer,
                "is_yes": is_yes,
                "trial_idx": trial_idx,
                "model": MODEL
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Trial failed for {prompt_name} trial {trial_idx}: {e}")
            return None
    
    async def test_all_prompts(self):
        """Test all prompt variations with multiple trials"""
        logger.info(f"Starting GPT-4.1 prompt testing with {TRIALS_PER_PROMPT} trials per prompt")
        logger.info(f"Testing {len(PROMPT_VARIATIONS)} prompt variations")
        logger.info(f"Test question: {TEST_QUESTION}")
        
        total_trials = len(PROMPT_VARIATIONS) * TRIALS_PER_PROMPT
        
        with tqdm(total=total_trials, desc="Testing prompts") as pbar:
            for prompt_name, prompt_template in PROMPT_VARIATIONS.items():
                logger.info(f"Testing prompt: {prompt_name}")
                
                for trial_idx in range(TRIALS_PER_PROMPT):
                    result = await self.run_single_trial(prompt_name, prompt_template, trial_idx)
                    
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
        output_dir = Path("gpt4_1_prompt_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        results_file = output_dir / "detailed_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved detailed results to {results_file}")
        
        # Create summary by prompt
        summary = df.groupby('prompt_name').agg({
            'is_yes': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        
        summary.columns = ['prompt_name', 'n_trials', 'n_yes', 'percent_yes', 'std_dev']
        summary['percent_yes'] = (summary['percent_yes'] * 100).round(1)
        summary['std_dev'] = summary['std_dev'].round(3)
        
        summary_file = output_dir / "prompt_summary.csv"
        summary.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        return summary
    
    def create_graph(self, summary_df: pd.DataFrame):
        """Create a graph showing is_yes percentage for each prompt"""
        try:
            # Create the plot
            plt.figure(figsize=(20, 12))
            
            # Sort by percent_yes for better visualization
            summary_df = summary_df.sort_values('percent_yes', ascending=False)
            
            # Create bars
            bars = plt.bar(range(len(summary_df)), summary_df['percent_yes'], 
                          color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
            
            # Customize the plot
            plt.title(f'GPT-4.1 Response Patterns Across {len(summary_df)} Prompt Variations\n'
                     f'Question: "{TEST_QUESTION}" | {TRIALS_PER_PROMPT} trials per prompt', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Prompt Variation', fontweight='bold', fontsize=12)
            plt.ylabel('% Yes Responses', fontweight='bold', fontsize=12)
            plt.ylim(0, 100)
            
            # Set x-axis labels
            plt.xticks(range(len(summary_df)), summary_df['prompt_name'], 
                      rotation=45, ha='right', fontsize=10)
            
            # Add value labels on bars
            for i, (bar, percent, trials) in enumerate(zip(bars, summary_df['percent_yes'], summary_df['n_trials'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{percent:.1f}%\n({trials} trials)', ha='center', va='bottom', fontsize=8)
            
            # Add grid
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            output_dir = Path("gpt4_1_prompt_test_results")
            plot_file = output_dir / "gpt4_1_prompt_variations_graph.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved graph to {plot_file}")
            
            # Also create a horizontal version for better readability
            plt.figure(figsize=(16, 20))
            
            # Horizontal bar chart
            y_pos = np.arange(len(summary_df))
            bars = plt.barh(y_pos, summary_df['percent_yes'], 
                           color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
            
            plt.title(f'GPT-4.1 Response Patterns (Horizontal View)\n'
                     f'Question: "{TEST_QUESTION}" | {TRIALS_PER_PROMPT} trials per prompt', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('% Yes Responses', fontweight='bold', fontsize=12)
            plt.ylabel('Prompt Variation', fontweight='bold', fontsize=12)
            plt.xlim(0, 100)
            
            # Set y-axis labels
            plt.yticks(y_pos, summary_df['prompt_name'], fontsize=10)
            
            # Add value labels on bars
            for i, (bar, percent, trials) in enumerate(zip(bars, summary_df['percent_yes'], summary_df['n_trials'])):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{percent:.1f}% ({trials} trials)', ha='left', va='center', fontsize=8)
            
            # Add grid
            plt.grid(True, alpha=0.3, axis='x')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the horizontal plot
            horizontal_plot_file = output_dir / "gpt4_1_prompt_variations_horizontal.png"
            plt.savefig(horizontal_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved horizontal graph to {horizontal_plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
    
    def generate_report(self, summary_df: pd.DataFrame):
        """Generate a text report of the results"""
        try:
            output_dir = Path("gpt4_1_prompt_test_results")
            report_file = output_dir / "test_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("GPT-4.1 PROMPT VARIATION TEST REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Test Question: {TEST_QUESTION}\n")
                f.write(f"Model: {MODEL}\n")
                f.write(f"Trials per prompt: {TRIALS_PER_PROMPT}\n")
                f.write(f"Total prompt variations: {len(PROMPT_VARIATIONS)}\n")
                f.write(f"Total trials: {len(self.results)}\n")
                f.write(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("RESULTS SUMMARY:\n")
                f.write("-" * 20 + "\n")
                
                # Sort by percent_yes
                sorted_summary = summary_df.sort_values('percent_yes', ascending=False)
                
                for _, row in sorted_summary.iterrows():
                    f.write(f"{row['prompt_name']}: {row['percent_yes']:.1f}% Yes ({row['n_trials']} trials)\n")
                
                f.write(f"\nSTATISTICS:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Highest % Yes: {sorted_summary['percent_yes'].max():.1f}%\n")
                f.write(f"Lowest % Yes: {sorted_summary['percent_yes'].min():.1f}%\n")
                f.write(f"Mean % Yes: {sorted_summary['percent_yes'].mean():.1f}%\n")
                f.write(f"Standard Deviation: {sorted_summary['percent_yes'].std():.1f}%\n")
                
                # Find most and least effective prompts
                most_yes = sorted_summary.iloc[0]
                least_yes = sorted_summary.iloc[-1]
                
                f.write(f"\nMOST EFFECTIVE PROMPT (Highest % Yes):\n")
                f.write(f"  {most_yes['prompt_name']}: {most_yes['percent_yes']:.1f}%\n")
                
                f.write(f"\nLEAST EFFECTIVE PROMPT (Lowest % Yes):\n")
                f.write(f"  {least_yes['prompt_name']}: {least_yes['percent_yes']:.1f}%\n")
            
            logger.info(f"Generated report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


async def main():
    """Main function to run the prompt testing"""
    logger.info("Starting GPT-4.1 prompt variation test...")
    
    # Create tester
    tester = GPT41PromptTester()
    
    try:
        # Run all tests
        await tester.test_all_prompts()
        
        # Save results
        summary = tester.save_results()
        
        if summary is not None:
            # Create graphs
            tester.create_graph(summary)
            
            # Generate report
            tester.generate_report(summary)
            
            logger.info("Testing completed successfully!")
            logger.info(f"Results saved in: gpt4_1_prompt_test_results/")
            
            # Print summary to console
            print("\n" + "="*80)
            print("GPT-4.1 PROMPT TESTING COMPLETED")
            print("="*80)
            print(f"Question tested: {TEST_QUESTION}")
            print(f"Total prompts tested: {len(PROMPT_VARIATIONS)}")
            print(f"Trials per prompt: {TRIALS_PER_PROMPT}")
            print(f"Total trials: {len(tester.results)}")
            print(f"\nTop 5 prompts by % Yes:")
            
            sorted_summary = summary.sort_values('percent_yes', ascending=False)
            for i, (_, row) in enumerate(sorted_summary.head().iterrows()):
                print(f"{i+1}. {row['prompt_name']}: {row['percent_yes']:.1f}%")
            
            print(f"\nResults saved in: gpt4_1_prompt_test_results/")
            
        else:
            logger.error("No results to process")
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
