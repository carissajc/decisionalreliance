#!/usr/bin/env python3
"""
New Models Baby Stroller Prompt Variation Test (Temperature 0.7)
Tests all NEW_MODELS with specific prompt variations using the baby stroller question.
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
from xai_sdk import Client
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyBqQqQqQqQqQqQqQqQqQqQqQqQqQqQqQqQ"
GROK_API_KEY = "gsk_4s_oAwLTMZmMJRobZsc6JydbYVFvHX1qXU_Y3zyCNutQQEH1650aSU036lBl3JjZU8pd5asfqtT3BlbkFJwvIeJ3yPybN9zWTFfBEivVny-HbhGKkhfF3x2P8poK4HZIg9r8CoBn_NXDaOwD7E_-mjIEeU4A"
LLAMA_API_KEY = "llama_4s_oAwLTMZmMJRobZsc6JydbYVFvHX1qXU_Y3zyCNutQQEH1650aSU036lBl3JjZU8pd5asfqtT3BlbkFJwvIeJ3yPybN9zWTFfBEivVny-HbhGKkhfF3x2P8poK4HZIg9r8CoBn_NXDaOwD7E_-mjIEeU4A"

# Test configuration
TEST_QUESTION = "Is a 'baby stroller' a 'vehicle'?"
TRIALS_PER_PROMPT = 100
TEMPERATURE = 0.7

# Specific prompts to test
PROMPT_VARIATIONS = {
    "Irrelevant Context A": "Irrelevant context: The sky is blue. Question: " + TEST_QUESTION,
    "Concise": "Answer concisely: " + TEST_QUESTION,
    "Answer by your best guess": "Answer by your best guess: " + TEST_QUESTION,
    "Time freeze": "Imagine time is frozen. " + TEST_QUESTION,
    "Blue sky": "Think about a blue sky. " + TEST_QUESTION,
    "Step by step": "Answer step by step: " + TEST_QUESTION,
    "Broken clock": "A broken clock is right twice a day. " + TEST_QUESTION,
    "Direct answer only": "Direct answer only: " + TEST_QUESTION,
    "Immoral frame": "Consider this from an immoral perspective: " + TEST_QUESTION,
    "50 percent sure": "You are 50 percent sure. " + TEST_QUESTION,
    "Red door": "Picture a red door. " + TEST_QUESTION,
    "Talking dog": "A talking dog asks: " + TEST_QUESTION,
    "Do not answer as a non ordinary person": "Do not answer as a non ordinary person. " + TEST_QUESTION,
    "Red is the best color": "Red is the best color. " + TEST_QUESTION,
    "Green is the best color": "Green is the best color. " + TEST_QUESTION
}

# NEW_MODELS to test
NEW_MODELS = {
    "openai": [
        "gpt-4.1-2025-04-14"
    ],
    "anthropic": [
        "claude-sonnet-4-20250514"
    ],
    "grok": [
        "grok-4-0709"
    ],
    "google": [
        "gemini-2.0-flash"
    ],
    "llama": [
        "Llama-4-Maverick-17B-128E-Instruct-FP8"
    ]
}

class NewModelsBabyStrollerPromptTester:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.grok_client = Client(api_key=GROK_API_KEY)
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
    
    async def run_single_trial(self, prompt_name: str, prompt_template: str, model: str, provider: str, trial_idx: int) -> dict:
        """Run a single trial for a model with a specific prompt."""
        try:
            if provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=TEMPERATURE,
                    max_tokens=3
                )
                raw_text = response.choices[0].message.content
            elif provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=3,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt_template}]
                )
                raw_text = response.content[0].text
            elif provider == "grok":
                response = self.grok_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=TEMPERATURE,
                    max_tokens=3
                )
                raw_text = response.choices[0].message.content
            elif provider == "google":
                # Use aiohttp for Google Gemini API
                async with aiohttp.ClientSession() as session:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt_template}]}],
                        "generationConfig": {
                            "temperature": TEMPERATURE,
                            "maxOutputTokens": 3
                        }
                    }
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            raise Exception(f"Google API error: {resp.status}")
            elif provider == "llama":
                # Use aiohttp for Llama Cloud API
                async with aiohttp.ClientSession() as session:
                    url = "https://api.llama-api.com/chat/completions"
                    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_template}],
                        "temperature": TEMPERATURE,
                        "max_tokens": 3
                    }
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            raw_text = data["choices"][0]["message"]["content"]
                        else:
                            raise Exception(f"Llama API error: {resp.status}")
            else:
                return None
            
            extracted_answer = self._normalize_response(raw_text)
            
            return {
                'prompt_name': prompt_name,
                'prompt_template': prompt_template,
                'model': model,
                'provider': provider,
                'trial': trial_idx,
                'question': TEST_QUESTION,
                'raw_text': raw_text,
                'extracted_answer': extracted_answer,
                'temperature': TEMPERATURE
            }
            
        except Exception as e:
            logger.error(f"Error in trial {trial_idx} for {model} with prompt '{prompt_name}': {e}")
            return None
    
    async def test_all_prompts(self):
        """Test all prompt variations with all new models."""
        total_trials = len(PROMPT_VARIATIONS) * sum(len(models) for models in NEW_MODELS.values()) * TRIALS_PER_PROMPT
        
        with tqdm(total=total_trials, desc="Testing new models with prompts") as pbar:
            for prompt_name, prompt_template in PROMPT_VARIATIONS.items():
                logger.info(f"Testing prompt: {prompt_name}")
                
                for provider, models in NEW_MODELS.items():
                    for model in models:
                        logger.info(f"Testing model: {model}")
                        
                        for trial_idx in range(TRIALS_PER_PROMPT):
                            result = await self.run_single_trial(prompt_name, prompt_template, model, provider, trial_idx)
                            if result:
                                self.results.append(result)
                            pbar.update(1)
                            
                            # Small delay to avoid rate limits
                            await asyncio.sleep(0.1)
        
        logger.info(f"Completed testing. Total results: {len(self.results)}")
    
    def save_results(self):
        """Save results to CSV files."""
        output_dir = Path("new_models_baby_stroller_prompt_test_results_temp_0.7")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df_detailed = pd.DataFrame(self.results)
        detailed_path = output_dir / "detailed_results.csv"
        df_detailed.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to {detailed_path}")
        
        # Create summary
        summary_data = []
        for prompt_name in PROMPT_VARIATIONS.keys():
            for provider, models in NEW_MODELS.items():
                for model in models:
                    model_prompt_results = [r for r in self.results if r['model'] == model and r['prompt_name'] == prompt_name]
                    n_trials = len(model_prompt_results)
                    n_yes = len([r for r in model_prompt_results if r['extracted_answer'] == 'Yes'])
                    percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
                    
                    summary_data.append({
                        'prompt_name': prompt_name,
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
        """Create horizontal bar chart of results."""
        # Group by prompt and calculate average Yes percentage across models
        prompt_stats = {}
        for prompt_name in PROMPT_VARIATIONS.keys():
            prompt_results = [r for r in self.results if r['prompt_name'] == prompt_name]
            n_trials = len(prompt_results)
            n_yes = len([r for r in prompt_results if r['extracted_answer'] == 'Yes'])
            percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
            prompt_stats[prompt_name] = percent_yes
        
        # Sort by Yes percentage
        sorted_prompts = sorted(prompt_stats.items(), key=lambda x: x[1], reverse=True)
        prompt_names = [item[0] for item in sorted_prompts]
        yes_percentages = [item[1] for item in sorted_prompts]
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bars
        bars = plt.barh(range(len(prompt_names)), yes_percentages, color='skyblue', edgecolor='navy')
        
        # Customize the plot
        plt.title(f'New Models Baby Stroller Prompt Variations Test (Temperature {TEMPERATURE})\nAverage "Yes" Percentage by Prompt', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Percentage "Yes" Responses', fontsize=12, fontweight='bold')
        plt.ylabel('Prompt Variation', fontsize=12, fontweight='bold')
        
        # Set y-axis labels
        plt.yticks(range(len(prompt_names)), prompt_names)
        
        # Add value labels on bars
        for i, (bar, percentage) in enumerate(zip(bars, yes_percentages)):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{percentage:.1f}%', ha='left', va='center', fontweight='bold')
        
        # Set x-axis limits
        plt.xlim(0, 105)
        
        # Add grid
        plt.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        graph_path = output_dir / f"new_models_baby_stroller_prompt_variations_horizontal_temp_{TEMPERATURE}.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved graph to {graph_path}")
    
    def generate_report(self, output_dir: Path):
        """Generate a text report."""
        # Calculate summary statistics
        total_trials = len(self.results)
        successful_trials = len([r for r in self.results if r['extracted_answer'] != 'Unknown'])
        
        # Top 5 prompts by Yes percentage
        prompt_stats = {}
        for prompt_name in PROMPT_VARIATIONS.keys():
            prompt_results = [r for r in self.results if r['prompt_name'] == prompt_name]
            n_trials = len(prompt_results)
            n_yes = len([r for r in prompt_results if r['extracted_answer'] == 'Yes'])
            percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
            prompt_stats[prompt_name] = percent_yes
        
        top_prompts = sorted(prompt_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report
        report_content = f"""NEW MODELS BABY STROLLER PROMPT VARIATIONS TEST REPORT (TEMPERATURE {TEMPERATURE})

Question tested: {TEST_QUESTION}
Temperature: {TEMPERATURE}
Total prompts tested: {len(PROMPT_VARIATIONS)}
Total models tested: {sum(len(models) for models in NEW_MODELS.values())}
Trials per prompt per model: {TRIALS_PER_PROMPT}
Total trials: {total_trials}
Successful trials: {successful_trials}

Top 5 prompts by Yes percentage:"""
        
        for i, (prompt_name, percent_yes) in enumerate(top_prompts, 1):
            report_content += f"\n{i}. {prompt_name}: {percent_yes:.1f}% Yes"
        
        report_content += f"\n\nResults saved in: {output_dir}/"
        
        # Save report
        report_path = output_dir / f"test_report_temp_{TEMPERATURE}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated report: {report_path}")
        print(report_content)

async def main():
    """Main function."""
    tester = NewModelsBabyStrollerPromptTester()
    
    print(f"Starting New Models Baby Stroller Prompt Variations Test (Temperature {TEMPERATURE})")
    print(f"Question: {TEST_QUESTION}")
    print(f"Trials per prompt per model: {TRIALS_PER_PROMPT}")
    print(f"Total prompts: {len(PROMPT_VARIATIONS)}")
    print("=" * 80)
    
    await tester.test_all_prompts()
    
    output_dir = tester.save_results()
    tester.create_graph(output_dir)
    tester.generate_report(output_dir)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
