#!/usr/bin/env python3
"""
Continue Full LLM Experiment - Historical and New Models
This script continues the experiment from where it left off, incorporating all existing data
and running both historical and new models for the remaining questions.
"""

import asyncio
import json
import logging
import os
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import openai
import anthropic
from google.generativeai import GenerativeModel
from xai_sdk import Client
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import configuration
from experiment_config import (
    API_KEYS, HISTORICAL_MODELS, NEW_MODELS, PROMPT_VARIATIONS, 
    TEMPERATURES, TRIALS_PER_COMBINATION, FULL_QUESTIONS,
    OUTPUT_DIR, FULL_EXPERIMENT_DIR, CHARTS_DIR, LOGS_DIR, LOG_FILES,
    SYSTEM_MESSAGES
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a file handler for detailed logging
file_handler = logging.FileHandler(LOGS_DIR / "continue_experiment.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ContinueFullExperiment:
    def __init__(self):
        self.clients = {}
        self.existing_data = {}
        self.completed_questions = set()
        self._initialize_clients()
        self._load_existing_data()
        
    def _initialize_clients(self):
        """Initialize API clients for all providers"""
        try:
            # OpenAI
            if API_KEYS.get("openai"):
                self.clients["openai"] = openai.OpenAI(api_key=API_KEYS["openai"])
                logger.info("OpenAI client initialized")
            
            # Anthropic
            if API_KEYS.get("anthropic"):
                self.clients["anthropic"] = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
                logger.info("Anthropic client initialized")
            
            # Google (using aiohttp for direct API calls)
            if API_KEYS.get("google"):
                self.clients["google"] = API_KEYS["google"]
                logger.info("Google API key loaded")
            
            # Grok
            if API_KEYS.get("grok"):
                self.clients["grok"] = Client(API_KEYS["grok"])
                logger.info("Grok client initialized")
            
            # Llama (using aiohttp for direct API calls)
            if API_KEYS.get("llama"):
                self.clients["llama"] = API_KEYS["llama"]
                logger.info("Llama API key loaded")
                
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise
    
    def _load_existing_data(self):
        """Load all existing experiment data to understand what's already completed"""
        logger.info("Loading existing experiment data...")
        
        for question_dir in FULL_EXPERIMENT_DIR.iterdir():
            if question_dir.is_dir() and question_dir.name.startswith("question_"):
                question_num = int(question_dir.name.split("_")[1])
                self.completed_questions.add(question_num)
                
                # Load summary data
                summary_file = question_dir / f"summary_question_{question_num}.csv"
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file)
                        self.existing_data[question_num] = df
                        logger.info(f"Loaded data for question {question_num}: {len(df)} rows")
                    except Exception as e:
                        logger.error(f"Error loading summary for question {question_num}: {e}")
        
        logger.info(f"Found {len(self.completed_questions)} completed questions: {sorted(self.completed_questions)}")
    
    def _load_questions(self) -> List[Dict]:
        """Load questions from hypotheticals.xlsx"""
        try:
            df = pd.read_excel("data/hypotheticals.xlsx")
            
            # Drop specified columns and rename
            columns_to_drop = ['Case Inspiration', 'Other Considerations', 'Other Thoughts']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Rename columns
            df = df.rename(columns={
                'Question': 'question',
                'Answer': 'answer',
                'Category': 'category'
            })
            
            # Convert to list of dictionaries
            questions = df.to_dict('records')
            logger.info(f"Loaded {len(questions)} questions from hypotheticals.xlsx")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise
    
    async def _call_openai(self, model: str, prompt: str, temperature: float) -> str:
        """Call OpenAI API"""
        try:
            if "gpt-4.1" in model:
                response = self.clients["openai"].chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_MESSAGES["openai"]},
                             {"role": "user", "content": prompt}],
                    max_completion_tokens=3,
                    temperature=temperature
                )
            else:
                response = self.clients["openai"].chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_MESSAGES["openai"]},
                             {"role": "user", "content": prompt}],
                    max_tokens=3,
                    temperature=temperature
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error for {model}: {e}")
            raise
    
    async def _call_anthropic(self, model: str, prompt: str, temperature: float) -> str:
        """Call Anthropic API"""
        try:
            response = self.clients["anthropic"].messages.create(
                model=model,
                max_tokens=3,
                temperature=temperature,
                system=SYSTEM_MESSAGES["anthropic"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error for {model}: {e}")
            raise
    
    async def _call_google(self, model: str, prompt: str, temperature: float) -> str:
        """Call Google Gemini API using aiohttp"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 3,
                        "temperature": temperature
                    }
                }
                
                # Add system message to the prompt
                full_prompt = f"{SYSTEM_MESSAGES['google']}\n\n{prompt}"
                data["contents"][0]["parts"][0]["text"] = full_prompt
                
                async with session.post(url, headers=headers, json=data, 
                                      params={"key": self.clients["google"]}) as response:
                    if response.status == 500:
                        # Retry with exponential backoff for 500 errors
                        for attempt in range(3):
                            await asyncio.sleep(2 ** attempt)
                            async with session.post(url, headers=headers, json=data,
                                                  params={"key": self.clients["google"]}) as retry_response:
                                if retry_response.status == 200:
                                    result = await retry_response.json()
                                    break
                        else:
                            raise Exception("Failed after 3 retries for 500 error")
                    else:
                        result = await response.json()
                    
                    if "candidates" in result and len(result["candidates"]) > 0:
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise Exception(f"Unexpected response format: {result}")
                        
        except Exception as e:
            logger.error(f"Google API error for {model}: {e}")
            raise
    
    async def _call_grok(self, model: str, prompt: str, temperature: float) -> str:
        """Call Grok API"""
        try:
            # Add system message to the prompt
            full_prompt = f"{SYSTEM_MESSAGES['grok']}\n\n{prompt}"
            
            response = self.clients["grok"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=3,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Grok API error for {model}: {e}")
            raise
    
    async def _call_llama(self, model: str, prompt: str, temperature: float) -> str:
        """Call Llama Cloud API using aiohttp"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.llama-cloud.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.clients['llama']}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGES["llama"]},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 3,
                    "temperature": temperature
                }
                
                async with session.post(url, headers=headers, json=data) as response:
                    result = await response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["completion_message"]["content"]["text"]
                    else:
                        raise Exception(f"Unexpected response format: {result}")
                        
        except Exception as e:
            logger.error(f"Llama API error for {model}: {e}")
            raise
    
    def _normalize_response(self, text: str) -> Tuple[str, str, Optional[int]]:
        """Normalize response and extract Yes/No"""
        raw_text = text.strip() if text else ""
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
    
    async def run_single_trial(self, provider: str, model: str, prompt: str, 
                              temperature: float, prompt_variation: str = "normal") -> Optional[Dict]:
        """Run a single trial for a model"""
        try:
            if provider == "openai":
                response = await self._call_openai(model, prompt, temperature)
            elif provider == "anthropic":
                response = await self._call_anthropic(model, prompt, temperature)
            elif provider == "google":
                response = await self._call_google(model, prompt, temperature)
            elif provider == "grok":
                response = await self._call_grok(model, prompt, temperature)
            elif provider == "llama":
                response = await self._call_llama(model, prompt, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            raw_text, extracted_answer, is_yes = self._normalize_response(response)
            
            return {
                "provider": provider,
                "model": model,
                "prompt_variation": prompt_variation,
                "temperature": temperature,
                "raw_text": raw_text,
                "extracted_answer": extracted_answer,
                "is_yes": is_yes,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Trial failed for {provider}/{model}: {e}")
            return None
    
    async def run_experiment_for_question(self, question_data: Dict, question_num: int):
        """Run full experiment for a single question"""
        logger.info(f"Starting experiment for Question {question_num}: {question_data['question'][:100]}...")
        
        question_dir = FULL_EXPERIMENT_DIR / f"question_{question_num}"
        question_dir.mkdir(exist_ok=True)
        
        all_trials = []
        failed_trials = []
        
        # Run historical models (temperature variations only, no prompt variations)
        logger.info(f"Running historical models for question {question_num}...")
        for provider, models in HISTORICAL_MODELS.items():
            for model in models:
                for temperature in TEMPERATURES:
                    for trial in range(TRIALS_PER_COMBINATION):
                        prompt = question_data['question']
                        
                        result = await self.run_single_trial(
                            provider, model, prompt, temperature, "normal"
                        )
                        
                        if result:
                            all_trials.append(result)
                        else:
                            failed_trials.append({
                                "provider": provider,
                                "model": model,
                                "temperature": temperature,
                                "prompt_variation": "normal",
                                "trial": trial,
                                "error": "API call failed"
                            })
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
        
        # Run new models (full experimental protocol)
        logger.info(f"Running new models for question {question_num}...")
        for provider, models in NEW_MODELS.items():
            for model in models:
                for prompt_variation, prompt_template in PROMPT_VARIATIONS.items():
                    for temperature in TEMPERATURES:
                        for trial in range(TRIALS_PER_COMBINATION):
                            prompt = prompt_template.format(question=question_data['question'])
                            
                            result = await self.run_single_trial(
                                provider, model, prompt, temperature, prompt_variation
                            )
                            
                            if result:
                                all_trials.append(result)
                            else:
                                failed_trials.append({
                                    "provider": provider,
                                    "model": model,
                                    "temperature": temperature,
                                    "prompt_variation": prompt_variation,
                                    "trial": trial,
                                    "error": "API call failed"
                                })
                            
                            # Small delay to avoid rate limiting
                            await asyncio.sleep(0.1)
        
        # Save trial results
        if all_trials:
            trials_df = pd.DataFrame(all_trials)
            trials_file = question_dir / f"trials_question_{question_num}.csv"
            trials_df.to_csv(trials_file, index=False)
            logger.info(f"Saved {len(all_trials)} trials for question {question_num}")
        
        # Save failed trials
        if failed_trials:
            failed_df = pd.DataFrame(failed_trials)
            failed_file = question_dir / f"failed_trials_question_{question_num}.csv"
            failed_df.to_csv(failed_file, index=False)
            logger.warning(f"Recorded {len(failed_trials)} failed trials for question {question_num}")
        
        # Create and save summary
        self._create_question_summary(question_num, all_trials)
        
        # Generate charts
        self._generate_question_charts(question_num, question_data)
        
        logger.info(f"Completed experiment for question {question_num}")
    
    def _create_question_summary(self, question_num: int, trials: List[Dict]):
        """Create summary statistics for a question"""
        if not trials:
            logger.warning(f"No trials to summarize for question {question_num}")
            return
        
        df = pd.DataFrame(trials)
        
        # Group by model, prompt variation, and temperature
        summary_data = []
        
        for (model, prompt_var, temp), group in df.groupby(['model', 'prompt_variation', 'temperature']):
            n_trials = len(group)
            n_yes = group['is_yes'].sum() if group['is_yes'].notna().any() else 0
            percent_yes = (n_yes / n_trials * 100) if n_trials > 0 else 0
            std_dev = group['is_yes'].std() if group['is_yes'].notna().any() else 0
            
            # Determine provider from model
            provider = None
            for prov, models in {**HISTORICAL_MODELS, **NEW_MODELS}.items():
                if model in models:
                    provider = prov
                    break
            
            summary_data.append({
                'provider': provider,
                'model': model,
                'prompt_variation': prompt_var,
                'temperature': temp,
                'n_trials': n_trials,
                'n_yes': n_yes,
                'percent_yes': percent_yes,
                'std_dev': std_dev
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = FULL_EXPERIMENT_DIR / f"question_{question_num}" / f"summary_question_{question_num}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Created summary for question {question_num}: {len(summary_df)} combinations")
    
    def _generate_question_charts(self, question_num: int, question_data: Dict):
        """Generate charts for a single question"""
        try:
            # Load the summary data for this question
            summary_file = FULL_EXPERIMENT_DIR / f"question_{question_num}" / f"summary_question_{question_num}.csv"
            if not summary_file.exists():
                logger.warning(f"Summary file not found for question {question_num}, skipping charts")
                return
            
            df = pd.read_csv(summary_file)
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'Question {question_num}: {question_data["question"][:100]}...', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Chart 1: Historical Models (temperature variations)
            self._generate_chart_1_historical_models(ax1, df, question_data)
            
            # Chart 2: New Models (prompt variations)
            self._generate_chart_2_new_models_prompts(ax2, df, question_data)
            
            # Chart 3: New Models (temperature effects)
            self._generate_chart_3_new_models_temperatures(ax3, df, question_data)
            
            plt.tight_layout()
            
            # Save chart
            chart_file = CHARTS_DIR / f"question_{question_num}_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated charts for question {question_num}")
            
        except Exception as e:
            logger.error(f"Error generating charts for question {question_num}: {e}")
    
    def _generate_chart_1_historical_models(self, ax, df: pd.DataFrame, question_data: Dict):
        """Generate Chart 1: Historical Models with temperature variations"""
        # Filter for historical models only
        historical_models = []
        for provider, models in HISTORICAL_MODELS.items():
            historical_models.extend(models)
        
        historical_df = df[df['model'].isin(historical_models)].copy()
        
        if historical_df.empty:
            ax.text(0.5, 0.5, 'No historical model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 1: Historical Models\n% Yes Across All Trials', fontweight='bold', pad=20)
            return
        
        # Calculate average % Yes across all temperatures for each model
        model_stats = []
        for model in historical_models:
            model_data = historical_df[historical_df['model'] == model]
            if not model_data.empty:
                avg_percent = model_data['percent_yes'].mean()
                total_trials = model_data['n_trials'].sum()
                model_stats.append({
                    'model': model,
                    'percent_yes': avg_percent,
                    'total_trials': total_trials
                })
        
        if not model_stats:
            ax.text(0.5, 0.5, 'No historical model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 1: Historical Models\n% Yes Across All Trials', fontweight='bold', pad=20)
            return
        
        stats_df = pd.DataFrame(model_stats)
        
        # Use predefined order for historical models
        historical_model_order = [
            "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0125",
            "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
            "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-turbo-2024-04-09",
            "claude-3-5-haiku-20241022", "claude-3.5-haiku-latest",
            "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest", "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest"
        ]
        
        # Filter to only include models we have data for
        available_models = [m for m in historical_model_order if m in stats_df['model'].values]
        stats_df = stats_df[stats_df['model'].isin(available_models)]
        
        # Sort by the predefined order
        stats_df['model'] = pd.Categorical(stats_df['model'], categories=available_models, ordered=True)
        stats_df = stats_df.sort_values('model')
        
        bars = ax.bar(range(len(stats_df)), stats_df['percent_yes'], 
                     color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        
        ax.set_title('Chart 1: Historical Models\n% Yes Across All Trials', fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('% Yes', fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Set x-axis labels
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels([model.replace('gpt-', '').replace('claude-', '') 
                           for model in stats_df['model']], 
                          rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for i, (bar, percent, trials) in enumerate(zip(bars, stats_df['percent_yes'], stats_df['total_trials'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                   f'{percent:.1f}%\n({trials} trials)', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    def _generate_chart_2_new_models_prompts(self, ax, df: pd.DataFrame, question_data: Dict):
        """Generate Chart 2: New Models with prompt variations"""
        # Filter for new models only
        new_models = []
        for provider, models in NEW_MODELS.items():
            new_models.extend(models)
        
        new_df = df[df['model'].isin(new_models)].copy()
        
        if new_df.empty:
            ax.text(0.5, 0.5, 'No new model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 2: New Models\n% Yes by Prompt Variation', fontweight='bold', pad=20)
            return
        
        # Group by model and prompt variation
        prompt_data = []
        for model in new_models:
            model_data = new_df[new_df['model'] == model]
            if not model_data.empty:
                for prompt_var in PROMPT_VARIATIONS.keys():
                    prompt_data_for_var = model_data[model_data['prompt_variation'] == prompt_var]
                    if not prompt_data_for_var.empty:
                        avg_percent = prompt_data_for_var['percent_yes'].mean()
                        total_trials = prompt_data_for_var['n_trials'].sum()
                        prompt_data.append({
                            'model': model,
                            'prompt_variation': prompt_var,
                            'percent_yes': avg_percent,
                            'total_trials': total_trials
                        })
        
        if not prompt_data:
            ax.text(0.5, 0.5, 'No new model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 2: New Models\n% Yes by Prompt Variation', fontweight='bold', pad=20)
            return
        
        prompt_df = pd.DataFrame(prompt_data)
        
        # Create grouped bar chart
        models = prompt_df['model'].unique()
        prompt_vars = list(PROMPT_VARIATIONS.keys())
        
        x = np.arange(len(models))
        width = 0.25
        
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        
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
                             label=prompt_var.capitalize(), color=colors[i], alpha=0.7)
                
                # Add value labels
                for j, (bar, percent) in enumerate(zip(bars, aligned_data)):
                    if percent > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                               f'{percent:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Chart 2: New Models\n% Yes by Prompt Variation', fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('% Yes', fontweight='bold')
        ax.set_ylim(0, 100)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels([model.replace('gpt-', '').replace('claude-', '').replace('Llama-', '') 
                           for model in models], rotation=45, ha='right', fontsize=8)
        
        ax.legend(title='Prompt Variation')
        ax.grid(True, alpha=0.3)
    
    def _generate_chart_3_new_models_temperatures(self, ax, df: pd.DataFrame, question_data: Dict):
        """Generate Chart 3: New Models with temperature effects"""
        # Filter for new models only
        new_models = []
        for provider, models in NEW_MODELS.items():
            new_models.extend(models)
        
        new_df = df[df['model'].isin(new_models)].copy()
        
        if new_df.empty:
            ax.text(0.5, 0.5, 'No new model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 3: New Models\n% Yes by Temperature', fontweight='bold', pad=20)
            return
        
        # Group by model and temperature
        temp_data = []
        for model in new_models:
            model_data = new_df[new_df['model'] == model]
            if not model_data.empty:
                for temp in TEMPERATURES:
                    temp_data_for_temp = model_data[model_data['temperature'] == temp]
                    if not temp_data_for_temp.empty:
                        avg_percent = temp_data_for_temp['percent_yes'].mean()
                        total_trials = temp_data_for_temp['n_trials'].sum()
                        temp_data.append({
                            'model': model,
                            'temperature': temp,
                            'percent_yes': avg_percent,
                            'total_trials': total_trials
                        })
        
        if not temp_data:
            ax.text(0.5, 0.5, 'No new model data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Chart 3: New Models\n% Yes by Temperature', fontweight='bold', pad=20)
            return
        
        temp_df = pd.DataFrame(temp_data)
        
        # Create grouped bar chart
        models = temp_df['model'].unique()
        temps = TEMPERATURES
        
        x = np.arange(len(models))
        width = 0.25
        
        colors = ['gold', 'orange', 'red']
        
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
                             label=f'Temp={temp}', color=colors[i], alpha=0.7)
                
                # Add value labels
                for j, (bar, percent) in enumerate(zip(bars, aligned_data)):
                    if percent > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                               f'{percent:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Chart 3: New Models\n% Yes by Temperature', fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('% Yes', fontweight='bold')
        ax.set_ylim(0, 100)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels([model.replace('gpt-', '').replace('claude-', '').replace('Llama-', '') 
                           for model in models], rotation=45, ha='right', fontsize=8)
        
        ax.legend(title='Temperature')
        ax.grid(True, alpha=0.3)
    
    async def run_remaining_questions(self):
        """Run experiments for all remaining questions"""
        questions = self._load_questions()
        
        # Filter to only questions that haven't been completed
        remaining_questions = [q for i, q in enumerate(questions, 1) if i not in self.completed_questions]
        
        logger.info(f"Found {len(remaining_questions)} remaining questions to process")
        
        if not remaining_questions:
            logger.info("All questions have been completed!")
            return
        
        # Process remaining questions
        for i, question in enumerate(remaining_questions, 1):
            question_num = len(self.completed_questions) + i
            
            logger.info(f"Processing question {question_num}/{len(questions)} ({i}/{len(remaining_questions)} remaining)")
            
            try:
                await self.run_experiment_for_question(question, question_num)
                self.completed_questions.add(question_num)
                
                # Small delay between questions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing question {question_num}: {e}")
                continue
        
        logger.info("Completed all remaining questions!")
    
    def generate_final_summary(self):
        """Generate a final summary of all experiments"""
        logger.info("Generating final summary...")
        
        all_summaries = []
        
        for question_num in sorted(self.completed_questions):
            summary_file = FULL_EXPERIMENT_DIR / f"question_{question_num}" / f"summary_question_{question_num}.csv"
            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    df['question_number'] = question_num
                    all_summaries.append(df)
                except Exception as e:
                    logger.error(f"Error reading summary for question {question_num}: {e}")
        
        if all_summaries:
            combined_df = pd.concat(all_summaries, ignore_index=True)
            combined_file = OUTPUT_DIR / "combined_summary_all_questions.csv"
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Generated combined summary with {len(combined_df)} rows")
            
            # Generate data quality report
            self._generate_data_quality_report(combined_df)
        else:
            logger.warning("No summary data found to combine")

async def main():
    """Main function to run the continued experiment"""
    logger.info("Starting continued full experiment...")
    
    # Create experiment runner
    runner = ContinueFullExperiment()
    
    try:
        # Run remaining questions
        await runner.run_remaining_questions()
        
        # Generate final summary
        runner.generate_final_summary()
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    def _generate_data_quality_report(self, combined_df: pd.DataFrame):
        """Generate a data quality report"""
        try:
            report_file = OUTPUT_DIR / "data_quality_report_final.txt"
            
            with open(report_file, 'w') as f:
                f.write("FINAL DATA QUALITY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total questions processed: {len(self.completed_questions)}\n")
                f.write(f"Total data rows: {len(combined_df)}\n\n")
                
                # Provider summary
                f.write("PROVIDER SUMMARY:\n")
                f.write("-" * 20 + "\n")
                provider_counts = combined_df['provider'].value_counts()
                for provider, count in provider_counts.items():
                    f.write(f"{provider}: {count} rows\n")
                f.write("\n")
                
                # Model summary
                f.write("MODEL SUMMARY:\n")
                f.write("-" * 20 + "\n")
                model_counts = combined_df['model'].value_counts()
                for model, count in model_counts.items():
                    f.write(f"{model}: {count} rows\n")
                f.write("\n")
                
                # Trial count validation
                f.write("TRIAL COUNT VALIDATION:\n")
                f.write("-" * 25 + "\n")
                expected_trials = TRIALS_PER_COMBINATION
                trial_validation = combined_df.groupby(['model', 'prompt_variation', 'temperature'])['n_trials'].agg(['count', 'sum', 'mean'])
                
                for (model, prompt_var, temp), row in trial_validation.iterrows():
                    if row['sum'] != expected_trials:
                        f.write(f"WARNING: {model} {prompt_var} temp={temp}: {row['sum']} trials (expected {expected_trials})\n")
                    else:
                        f.write(f"✓ {model} {prompt_var} temp={temp}: {row['sum']} trials\n")
                f.write("\n")
                
                # Missing data check
                f.write("MISSING DATA CHECK:\n")
                f.write("-" * 20 + "\n")
                missing_data = combined_df[combined_df['n_trials'] == 0]
                if not missing_data.empty:
                    f.write(f"Found {len(missing_data)} combinations with 0 trials:\n")
                    for _, row in missing_data.iterrows():
                        f.write(f"  - {row['provider']}/{row['model']} {row['prompt_variation']} temp={row['temperature']}\n")
                else:
                    f.write("✓ No missing data found\n")
                f.write("\n")
                
                # Response validation
                f.write("RESPONSE VALIDATION:\n")
                f.write("-" * 20 + "\n")
                invalid_responses = combined_df[combined_df['percent_yes'].isna()]
                if not invalid_responses.empty:
                    f.write(f"Found {len(invalid_responses)} combinations with invalid responses\n")
                else:
                    f.write("✓ All responses are valid\n")
                f.write("\n")
                
                f.write("Report generated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            logger.info(f"Generated final data quality report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")

if __name__ == "__main__":
    asyncio.run(main())
