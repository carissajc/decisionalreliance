#!/usr/bin/env python3
"""
New Models Meal Reimbursement Prompt Variation Test (Temperature 0.7)
Tests all NEW_MODELS with specific prompt variations using the meal reimbursement question.
Includes Grok and Gemini models with updated API keys.
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
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GROK_API_KEY = "xai-a7AXIivfuo0BZrBUQLZluKzRoacD1to2ILecv7PJuWjdv9G13Yzjs2iWhAklCtRoUKalJsvNt3gHRBY3"
GEMINI_API_KEY = "AIzaSyD_dPLqEj6YA4t4vhsKuI4snEIPywPWxzY"

# Test configuration
TEST_QUESTION = "Is meal reimbursement a type of money remuneration?"
TRIALS_PER_PROMPT = 100
TEMPERATURE = 0.7
MAX_TOKENS = 20  # Increased from 3 to 20 to prevent truncation

# STRICT prompts that force Yes/No first
PROMPT_VARIATIONS = {
    "Irrelevant Context A": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nIrrelevant context: The sky is blue. Question: {TEST_QUESTION}\n\nAnswer:",
    "Concise": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nAnswer concisely: {TEST_QUESTION}\n\nAnswer:",
    "Answer by your best guess": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nAnswer by your best guess: {TEST_QUESTION}\n\nAnswer:",
    "Time freeze": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nImagine time is frozen. {TEST_QUESTION}\n\nAnswer:",
    "Blue sky": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nThink about a blue sky. {TEST_QUESTION}\n\nAnswer:",
    "Step by step": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nAnswer step by step: {TEST_QUESTION}\n\nAnswer:",
    "Broken clock": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nA broken clock is right twice a day. {TEST_QUESTION}\n\nAnswer:",
    "Direct answer only": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nDirect answer only: {TEST_QUESTION}\n\nAnswer:",
    "Immoral frame": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nConsider this from an immoral perspective: {TEST_QUESTION}\n\nAnswer:",
    "50 percent sure": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nYou are 50 percent sure. {TEST_QUESTION}\n\nAnswer:",
    "Red door": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nPicture a red door. {TEST_QUESTION}\n\nAnswer:",
    "Talking dog": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nA talking dog asks: {TEST_QUESTION}\n\nAnswer:",
    "Do not answer as a non ordinary person": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nDo not answer as a non ordinary person. {TEST_QUESTION}\n\nAnswer:",
    "Red is the best color": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nRed is the best color. {TEST_QUESTION}\n\nAnswer:",
    "Green is the best color": f"You must answer with ONLY 'Yes' or 'No' first, then provide explanation if needed.\n\nGreen is the best color. {TEST_QUESTION}\n\nAnswer:"
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
        "llama-3.1-8b-instruct"
    ]
}

class NewModelsMealReimbursementPromptTester:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.grok_client = Client(api_key=GROK_API_KEY)
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
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
        
        return 'Unknown'
    
    async def run_single_trial(self, provider: str, model: str, prompt_variation: str, trial_idx: int) -> dict:
        """Run a single trial for a model."""
        try:
            prompt = PROMPT_VARIATIONS[prompt_variation]
            
            if provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                raw_text = response.choices[0].message.content
                
            elif provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_text = response.content[0].text
                
            elif provider == "grok":
                response = self.grok_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                raw_text = response.choices[0].message.content
                
            elif provider == "google":
                # Gemini API call
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS
                    )
                )
                raw_text = response.text
                
            elif provider == "llama":
                # Llama API call using aiohttp
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {GROK_API_KEY}",  # Using Grok API key for Llama
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS
                    }
                    async with session.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            raw_text = result["choices"][0]["message"]["content"]
                        else:
                            raise Exception(f"Llama API error: {response.status}")
            else:
                logger.error(f"Unknown provider: {provider}")
                return None

            extracted_answer = self._normalize_response(raw_text)

            result = {
                'provider': provider,
                'model': model,
                'prompt_variation': prompt_variation,
                'trial': trial_idx,
                'question': TEST_QUESTION,
                'prompt': prompt,
                'raw_text': raw_text,
                'extracted_answer': extracted_answer,
                'temperature': TEMPERATURE,
                'timestamp': time.time()
            }
            return result

        except Exception as e:
            logger.error(f"Error in trial {trial_idx} for {provider}/{model}: {e}")
            return None

    async def test_all_prompt_variations(self):
        """Test all prompt variations for all models."""
        logger.info(f"Starting Meal Reimbursement test with {TRIALS_PER_PROMPT} trials per prompt variation")
        
        total_trials = sum(len(models) for models in NEW_MODELS.values()) * len(PROMPT_VARIATIONS) * TRIALS_PER_PROMPT
        
        with tqdm(total=total_trials, desc="Testing prompt variations") as pbar:
            for provider, models in NEW_MODELS.items():
                for model in models:
                    logger.info(f"Testing {provider}/{model}")
                    
                    for prompt_variation in PROMPT_VARIATIONS:
                        for trial_idx in range(TRIALS_PER_PROMPT):
                            result = await self.run_single_trial(provider, model, prompt_variation, trial_idx)
                            
                            if result:
                                self.results.append(result)
                            
                            pbar.update(1)
                            
                            # Rate limiting
                            await asyncio.sleep(0.1)
        
        logger.info(f"Completed {len(self.results)} trials")

    def save_results(self) -> Path:
        """Save results to CSV files."""
        try:
            output_dir = Path("meal_reimbursement_prompt_test_results_temp_0.7")
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            df_detailed = pd.DataFrame(self.results)
            detailed_file = output_dir / "detailed_results.csv"
            df_detailed.to_csv(detailed_file, index=False)
            logger.info(f"Saved detailed results to {detailed_file}")
            
            # Save model summary
            model_summary = []
            for provider, models in NEW_MODELS.items():
                for model in models:
                    model_results = [r for r in self.results if r['model'] == model]
                    if model_results:
                        yes_count = len([r for r in model_results if r['extracted_answer'] == 'Yes'])
                        total_trials = len(model_results)
                        percent_yes = (yes_count / total_trials) if total_trials > 0 else 0
                        model_summary.append({
                            'provider': provider, 'model': model, 'n_trials': total_trials,
                            'n_yes': yes_count, 'percent_yes': percent_yes
                        })
            
            df_model_summary = pd.DataFrame(model_summary)
            model_summary_file = output_dir / "model_summary.csv"
            df_model_summary.to_csv(model_summary_file, index=False)
            logger.info(f"Saved model summary to {model_summary_file}")
            
            # Save prompt variation summary
            prompt_summary = []
            for prompt_variation in PROMPT_VARIATIONS:
                prompt_results = [r for r in self.results if r['prompt_variation'] == prompt_variation]
                if prompt_results:
                    yes_count = len([r for r in prompt_results if r['extracted_answer'] == 'Yes'])
                    total_trials = len(prompt_results)
                    percent_yes = (yes_count / total_trials) if total_trials > 0 else 0
                    prompt_summary.append({
                        'prompt_name': prompt_variation, 'n_trials': total_trials,
                        'n_yes': yes_count, 'percent_yes': percent_yes
                    })
            
            df_prompt_summary = pd.DataFrame(prompt_summary)
            prompt_summary_file = output_dir / "prompt_summary.csv"
            df_prompt_summary.to_csv(prompt_summary_file, index=False)
            logger.info(f"Saved prompt summary to {prompt_summary_file}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None

    def create_graph(self, output_dir: Path):
        """Create a horizontal bar chart showing prompt variation results."""
        try:
            prompt_summary_path = output_dir / "prompt_summary.csv"
            df_summary = pd.read_csv(prompt_summary_path)
            
            # Sort by percent_yes for better visualization
            df_summary = df_summary.sort_values('percent_yes', ascending=True)
            
            plt.figure(figsize=(16, 10))
            bars = plt.barh(range(len(df_summary)), df_summary['percent_yes'])
            
            # Add title and labels
            plt.title(f'Is meal reimbursement a type of money remuneration? (Temperature {TEMPERATURE})\n'
                      f'Total Runs: {len(self.results):,} | Models: {", ".join([model for models in NEW_MODELS.values() for model in models])}',
                      fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Percentage "Yes"', fontsize=12, fontweight='bold')
            plt.ylabel('Prompt Variation', fontsize=12, fontweight='bold')
            
            # Set y-axis labels
            plt.yticks(range(len(df_summary)), df_summary['prompt_name'])
            
            # Add value labels on bars
            for i, (bar, row) in enumerate(zip(bars, df_summary.itertuples())):
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%', ha='left', va='center', fontweight='bold')
            
            plt.xlim(0, 105)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            graph_path = output_dir / "meal_reimbursement_prompt_variations_horizontal.png"
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved graph to {graph_path}")
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")

    def generate_report(self, output_dir: Path):
        """Generate a text report."""
        try:
            total_trials = len(self.results)
            successful_trials = len([r for r in self.results if r['extracted_answer'] != 'Unknown'])
            
            # Top 5 prompt variations by Yes responses
            prompt_yes_counts = {}
            for prompt_variation in PROMPT_VARIATIONS:
                prompt_results = [r for r in self.results if r['prompt_variation'] == prompt_variation]
                yes_count = len([r for r in prompt_results if r['extracted_answer'] == 'Yes'])
                prompt_yes_counts[prompt_variation] = yes_count
            
            top_prompts = sorted(prompt_yes_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report_content = f"""NEW MODELS MEAL REIMBURSEMENT PROMPT VARIATION TEST REPORT (TEMPERATURE {TEMPERATURE})

Question tested: {TEST_QUESTION}
Temperature: {TEMPERATURE}
Total models tested: {sum(len(models) for models in NEW_MODELS.values())}
Prompt variations tested: {len(PROMPT_VARIATIONS)}
Trials per prompt variation: {TRIALS_PER_PROMPT}
Total trials: {total_trials}
Successful trials: {successful_trials}

Models used: {", ".join([model for models in NEW_MODELS.values() for model in models])}

Top 5 prompt variations by number of Yes responses:"""
            
            for i, (prompt_var, yes_count) in enumerate(top_prompts, 1):
                percent = (yes_count / (len(NEW_MODELS) * TRIALS_PER_PROMPT)) * 100
                report_content += f"\n{i}. {prompt_var}: {yes_count}/{len(NEW_MODELS) * TRIALS_PER_PROMPT} Yes ({percent:.1f}%)"
            
            report_content += f"\n\nResults saved in: {output_dir}/"
            
            report_path = output_dir / f"test_report_temp_{TEMPERATURE}.txt"
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Generated report: {report_path}")
            print(report_content)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")

async def main():
    """Main function."""
    tester = NewModelsMealReimbursementPromptTester()
    
    print(f"Starting New Models Meal Reimbursement Prompt Variation Test (Temperature {TEMPERATURE})")
    print(f"Question: {TEST_QUESTION}")
    print(f"Trials per prompt variation: {TRIALS_PER_PROMPT}")
    print("=" * 80)
    
    await tester.test_all_prompt_variations()
    
    output_dir = tester.save_results()
    tester.create_graph(output_dir)
    tester.generate_report(output_dir)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved in: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())




