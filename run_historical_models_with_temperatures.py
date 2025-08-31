#!/usr/bin/env python3
"""
Run Historical Models with Temperature Variations
This script runs only the historical models with different temperature settings
"""

import asyncio
import logging
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from experiment_config import *
from comprehensive_experiment_runner import ComprehensiveExperimentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalModelsTemperatureRunner:
    """Runner for historical models with temperature variations"""
    
    def __init__(self):
        self.results = []
        self.questions = []
        self.runner = None  # Will be initialized once
        self.temperatures = [0.0, 0.5, 1.0]
        self.trials_per_temp = 5  # 5 trials per temperature per model
        self._load_questions()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger('historical_models_temp')
        handler = logging.FileHandler(LOGS_DIR / "historical_models_temperature.log")
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _load_questions(self):
        """Load questions from hypotheticals.xlsx"""
        try:
            df = pd.read_excel(Path("data") / "hypotheticals.xlsx")
            
            # Clean column names and drop unwanted columns
            df.columns = df.columns.str.strip()
            df = df.drop(columns=['Case Inspiration', 'Other Considerations', 'Other Thoughts'], errors='ignore')
            
            # Rename columns to match our template
            df = df.rename(columns={
                'Question': 'question',
                'Subset': 'subset', 
                'Set': 'set',
                'Statute': 'statute'
            })
            
            # Clean up data
            df = df.dropna(subset=['question'])
            df = df.fillna('')
            
            # Load first 50 questions
            self.questions = df.head(50).to_dict('records')
            logger.info(f"Loaded {len(self.questions)} questions")
                
        except Exception as e:
            logger.error(f"Error loading hypotheticals.xlsx: {e}")
            self.questions = []
    
    async def _initialize_runner(self):
        """Initialize the experiment runner once"""
        if self.runner is None:
            self.runner = ComprehensiveExperimentRunner()
            logger.info("Initialized experiment runner")
    
    async def run_single_trial(self, question_data: Dict, model: str, temperature: float, trial_idx: int) -> Optional[Dict]:
        """Run a single trial for a historical model with specific temperature"""
        try:
            # Ensure runner is initialized
            await self._initialize_runner()
            
            # Determine provider
            provider = None
            for prov, models in HISTORICAL_MODELS.items():
                if model in models:
                    provider = prov
                    break
            
            if not provider:
                logger.error(f"Unknown provider for model {model}")
                return None
            
            # Create prompt (just the question, no variations)
            prompt = question_data['question']
            
            # Make API call with specified temperature
            response, api_info = await self.runner._call_api(provider, model, prompt, temperature)
            
            if response:
                # Parse response
                raw_text, extracted_answer, is_yes = self.runner._normalize_response(response)
                
                # Create result
                result = {
                    'question_index': question_data['question_index'],
                    'question': question_data['question'],
                    'question_label': question_data['question'],
                    'subset': question_data.get('subset', ''),
                    'set': question_data.get('set', ''),
                    'statute': question_data.get('statute', ''),
                    'provider': provider,
                    'model': model,
                    'prompt_variation': 'normal',  # Fixed as normal
                    'temperature': temperature,
                    'trial_idx': trial_idx,
                    'raw_text': raw_text,
                    'extracted_answer': extracted_answer,
                    'is_yes': is_yes,
                    'api_status': api_info.get('status', 'unknown'),
                    'response_time': api_info.get('response_time', 0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                return result
            else:
                logger.error(f"Empty response for {model} at temp {temperature}")
                return None
                
        except Exception as e:
            logger.error(f"Error in trial {model} at temp {temperature}: {e}")
            return None
    
    async def run_experiment_for_question(self, question_data: Dict, question_idx: int):
        """Run experiment for a single question with temperature variations"""
        logger.info(f"Starting historical models with temperatures for question {question_idx + 1}: {question_data['question'][:50]}...")
        
        question_results = []
        
        # Get all historical models
        all_models = []
        for provider, models in HISTORICAL_MODELS.items():
            all_models.extend(models)
        
        # Calculate total trials for this question (5 trials per temperature per model)
        total_trials = len(all_models) * len(self.temperatures) * self.trials_per_temp
        
        with tqdm(total=total_trials, desc=f"Question {question_idx + 1}") as pbar:
            for model in all_models:
                for temperature in self.temperatures:
                    for trial_idx in range(self.trials_per_temp):
                        result = await self.run_single_trial(
                            question_data, model, temperature, trial_idx
                        )
                        
                        if result:
                            question_results.append(result)
                            self.results.append(result)
                        
                        pbar.update(1)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
        
        # Save results for this question immediately
        self._save_question_results(question_data, question_results, question_idx)
        
        # Generate charts for this question immediately
        self._generate_charts_for_question(question_data, question_results, question_idx)
        
        logger.info(f"Completed question {question_idx + 1}. Results: {len(question_results)} trials")
    
    def _save_question_results(self, question_data: Dict, results: List[Dict], question_idx: int):
        """Save results for a specific question immediately"""
        if not results:
            return
        
        try:
            # Create question directory
            question_dir = FULL_EXPERIMENT_DIR / f"question_{question_idx + 1}"
            question_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            df = pd.DataFrame(results)
            results_file = question_dir / f"results_question_{question_idx + 1}.csv"
            df.to_csv(results_file, index=False)
            
            # Save summary
            summary = self._create_question_summary(results)
            summary_file = question_dir / f"summary_question_{question_idx + 1}.csv"
            summary.to_csv(summary_file, index=False)
            
            logger.info(f"Saved results for question {question_idx + 1} to {question_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results for question {question_idx + 1}: {e}")
    
    def _create_question_summary(self, results: List[Dict]) -> pd.DataFrame:
        """Create summary statistics for a question"""
        try:
            df = pd.DataFrame(results)
            
            summary = df.groupby(['provider', 'model', 'temperature']).agg({
                'is_yes': ['count', 'sum', 'mean', 'std']
            }).reset_index()
            
            summary.columns = ['provider', 'model', 'temperature', 'n_trials', 'n_yes', 'percent_yes', 'std_dev']
            summary['percent_yes'] = (summary['percent_yes'] * 100).round(1)
            summary['std_dev'] = summary['std_dev'].round(3)
            
            return summary
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return pd.DataFrame()
    
    def _generate_charts_for_question(self, question_data: Dict, results: List[Dict], question_idx: int):
        """Generate charts for a specific question immediately"""
        try:
            from chart_generator import ChartGenerator
            chart_gen = ChartGenerator()
            chart_gen.generate_question_charts(question_data, results, question_idx)
            logger.info(f"Generated charts for question {question_idx + 1}")
        except Exception as e:
            logger.error(f"Error generating charts for question {question_idx + 1}: {e}")
    
    async def run_full_experiment(self):
        """Run the complete experiment for historical models with temperatures"""
        logger.info(f"Starting historical models with temperature variations experiment")
        logger.info(f"Total questions: {len(self.questions)}")
        logger.info(f"Temperatures: {self.temperatures}")
        logger.info(f"Trials per temperature per model: {self.trials_per_temp}")
        
        start_time = time.time()
        
        for i, question_data in enumerate(self.questions):
            question_data['question_index'] = i + 1
            await self.run_experiment_for_question(question_data, i)
            
            # Progress update
            elapsed = time.time() - start_time
            remaining_questions = len(self.questions) - (i + 1)
            if remaining_questions > 0:
                avg_time_per_question = elapsed / (i + 1)
                estimated_remaining = remaining_questions * avg_time_per_question
                logger.info(f"Progress: {i + 1}/{len(self.questions)} questions completed. "
                           f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"Historical models with temperatures experiment completed in {total_time/60:.1f} minutes")
        logger.info(f"Total results collected: {len(self.results)}")

async def main():
    """Main entry point"""
    logger.info("Starting historical models with temperature variations experiment...")
    
    runner = HistoricalModelsTemperatureRunner()
    await runner.run_full_experiment()
    
    logger.info("Historical models with temperature variations experiment completed!")

if __name__ == "__main__":
    asyncio.run(main())

