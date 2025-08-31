#!/usr/bin/env python3
"""
Comprehensive LLM Yes/No Experiment Runner
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from experiment_config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILES["progress"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveExperimentRunner:
    """Comprehensive experiment runner for all models and prompt variations"""
    
    def __init__(self, trial_mode: bool = True):
        self.trial_mode = trial_mode
        self.clients = {}
        self.results = []
        self.current_question = 0
        self.questions = []
        
        # Initialize clients
        self._initialize_clients()
        
        # Load questions
        self._load_questions()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup detailed logging for different aspects"""
        # API calls logger
        self.api_logger = logging.getLogger('api_calls')
        api_handler = logging.FileHandler(LOG_FILES["api_calls"])
        api_handler.setLevel(logging.INFO)
        self.api_logger.addHandler(api_handler)
        self.api_logger.setLevel(logging.INFO)
        
        # Prompts and responses logger
        self.prompt_logger = logging.getLogger('prompts_responses')
        prompt_handler = logging.FileHandler(LOG_FILES["prompts_responses"])
        prompt_handler.setLevel(logging.INFO)
        self.prompt_logger.addHandler(prompt_handler)
        self.prompt_logger.setLevel(logging.INFO)
        
        # Errors logger
        self.error_logger = logging.getLogger('errors')
        error_handler = logging.FileHandler(LOG_FILES["errors"])
        error_handler.setLevel(logging.INFO)
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.INFO)
    
    def _initialize_clients(self):
        """Initialize API clients for all providers"""
        try:
            # OpenAI
            if "openai" in API_KEYS:
                from openai import OpenAI
                self.clients["openai"] = OpenAI(api_key=API_KEYS["openai"])
                logger.info("✓ OpenAI client initialized")
            
            # Anthropic
            if "anthropic" in API_KEYS:
                import anthropic
                self.clients["anthropic"] = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
                logger.info("✓ Anthropic client initialized")
            
            # Google
            if "google" in API_KEYS:
                import google.generativeai as genai
                genai.configure(api_key=API_KEYS["google"])
                self.clients["google"] = genai
                logger.info("✓ Google Gemini client initialized")
            
            # Grok
            if "grok" in API_KEYS:
                # Note: Grok API integration may need specific package
                self.clients["grok"] = {"api_key": API_KEYS["grok"]}
                logger.info("✓ Grok client initialized (basic)")
            
            # Llama
            if "llama" in API_KEYS:
                # Note: Llama API integration may need specific package
                self.clients["llama"] = {"api_key": API_KEYS["llama"]}
                logger.info("✓ Llama client initialized (basic)")
                
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Install missing packages with: pip install -r requirements.txt")
    
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
            
            # Limit questions based on mode
            if self.trial_mode:
                self.questions = df.head(TRIAL_QUESTIONS).to_dict('records')
                logger.info(f"Trial mode: Loaded {len(self.questions)} questions")
            else:
                self.questions = df.head(FULL_QUESTIONS).to_dict('records')
                logger.info(f"Full mode: Loaded {len(self.questions)} questions")
                
        except Exception as e:
            logger.error(f"Error loading hypotheticals.xlsx: {e}")
            # Fallback questions
            self.questions = [{
                'question': 'Is a tomato a vegetable?',
                'subset': 'Botanical classification',
                'set': 'Fruits vs Vegetables',
                'statute': 'In botanical terms, a tomato is classified as a fruit.'
            }]
    
    def _get_provider_for_model(self, model: str) -> str:
        """Determine which provider a model belongs to"""
        for provider, models in {**HISTORICAL_MODELS, **NEW_MODELS}.items():
            if model in models:
                return provider
        return "unknown"
    
    async def _call_api(self, provider: str, model: str, prompt: str, temperature: float) -> Tuple[str, Dict]:
        """Make API call with comprehensive error handling and logging"""
        start_time = time.time()
        
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
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Log API call
            self.api_logger.info(f"API_CALL|{provider}|{model}|{temperature}|{response_time:.2f}s|SUCCESS")
            
            return response, {"response_time": response_time, "status": "success"}
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            # Log error
            self.error_logger.error(f"API_ERROR|{provider}|{model}|{temperature}|{response_time:.2f}s|{str(e)}")
            self.api_logger.info(f"API_CALL|{provider}|{model}|{temperature}|{response_time:.2f}s|ERROR:{str(e)}")
            
            raise e
    
    async def _call_openai(self, model: str, prompt: str, temperature: float) -> str:
        """Make OpenAI API call"""
        client = self.clients["openai"]
        
        if model == "gpt-3.5-turbo-instruct":
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=3,
                temperature=temperature
            )
            return response.choices[0].text
        else:
            # Use max_completion_tokens for newer models like GPT-4.1
            if "gpt-4.1" in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGES["openai"]},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=3,
                    temperature=temperature
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGES["openai"]},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3,
                    temperature=temperature
                )
            return response.choices[0].message.content
    
    async def _call_anthropic(self, model: str, prompt: str, temperature: float) -> str:
        """Make Anthropic Claude API call"""
        client = self.clients["anthropic"]
        response = client.messages.create(
            model=model,
            max_tokens=3,
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_MESSAGES["anthropic"],
            temperature=temperature
        )
        return response.content[0].text if response.content else ""
    
    async def _call_google(self, model: str, prompt: str, temperature: float) -> str:
        """Make Google Gemini API call using the new endpoint with retry logic"""
        import aiohttp
        import json
        
        api_key = API_KEYS["google"]
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 3,
                "temperature": temperature
            }
        }
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Extract text from the new API response format
                            if 'candidates' in result and len(result['candidates']) > 0:
                                content = result['candidates'][0].get('content', {})
                                parts = content.get('parts', [])
                                if parts and len(parts) > 0:
                                    return parts[0].get('text', '')
                            return ''
                        else:
                            error_text = await response.text()
                            if "500" in error_text and attempt < max_retries - 1:
                                logger.warning(f"Google API 500 error, retrying {attempt + 1}/{max_retries}")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            raise Exception(f"Google API error: {response.status} - {error_text}")
                            
            except Exception as e:
                if "500" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Google API 500 error, retrying {attempt + 1}/{max_retries}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception(f"Google API error: {str(e)}")
    
    async def _call_grok(self, model: str, prompt: str, temperature: float) -> str:
        """Make Grok API call using xai-sdk"""
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user, system
            
            client = Client(
                api_key=API_KEYS["grok"],
                timeout=3600
            )
            
            chat = client.chat.create(model="grok-4")
            chat.append(system(SYSTEM_MESSAGES["grok"]))
            chat.append(user(prompt))
            
            response = chat.sample()
            return response.content
            
        except ImportError:
            raise Exception("xai-sdk not installed. Run: pip install xai-sdk")
        except Exception as e:
            raise Exception(f"Grok API error: {str(e)}")
    
    async def _call_llama(self, model: str, prompt: str, temperature: float) -> str:
        """Make Llama API call using aiohttp"""
        try:
            import aiohttp
            
            url = "https://api.llama.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEYS['llama']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGES["llama"]
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 3,
                "temperature": temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle the actual Llama API response format
                        if "completion_message" in result and "content" in result["completion_message"]:
                            content = result["completion_message"]["content"]
                            if isinstance(content, dict) and "text" in content:
                                return content["text"]
                            elif isinstance(content, str):
                                return content
                            else:
                                raise Exception(f"Unexpected content format: {content}")
                        else:
                            # Log the actual response for debugging
                            logger.warning(f"Llama API response missing completion_message: {result}")
                            raise Exception(f"Response missing completion_message field: {result}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Llama API HTTP error: {response.status} - {error_text}")
                        
        except ImportError:
            raise Exception("aiohttp not installed. Run: pip install aiohttp")
        except Exception as e:
            raise Exception(f"Llama API error: {str(e)}")
    
    def _normalize_response(self, text: str) -> Tuple[str, str, Optional[int]]:
        """Parse response to extract Yes/No and calculate is_yes"""
        # Keep the complete raw response exactly as returned
        raw_text = text.strip() if text else ""
        
        # Extract Yes/No from the beginning of the response
        # Look for "Yes" or "No" at the start (case-insensitive)
        if raw_text.lower().startswith("yes"):
            extracted_answer = "Yes"
            is_yes = 1
        elif raw_text.lower().startswith("no"):
            extracted_answer = "No"
            is_yes = 0
        else:
            extracted_answer = raw_text  # Keep original if no Yes/No found
            is_yes = None
            
        return raw_text, extracted_answer, is_yes
    
    async def run_single_trial(self, question_data: Dict, model: str, prompt_variation: str, 
                              temperature: float, trial_idx: int) -> Optional[Dict]:
        """Run a single trial with comprehensive logging"""
        provider = self._get_provider_for_model(model)
        
        # Build prompt
        prompt = PROMPT_VARIATIONS[prompt_variation].format(question=question_data['question'])
        
        # Log prompt
        self.prompt_logger.info(f"PROMPT|{question_data.get('question_index', 0)}|{model}|{prompt_variation}|{temperature}|{trial_idx}|{prompt}")
        
        try:
            # Make API call
            raw_response, api_metadata = await self._call_api(provider, model, prompt, temperature)
            
            # Parse response to extract Yes/No and keep raw text
            raw_text, extracted_answer, is_yes = self._normalize_response(raw_response)
            
            # Log response with raw text
            self.prompt_logger.info(f"RESPONSE|{question_data.get('question_index', 0)}|{model}|{prompt_variation}|{temperature}|{trial_idx}|{raw_response}|{extracted_answer}")
            
            result = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question_index": question_data.get('question_index', 0),
                "question_label": question_data['question'],
                "provider": provider,
                "model": model,
                "prompt_variation": prompt_variation,
                "temperature": temperature,
                "trial_idx": trial_idx,
                "raw_text": raw_response,
                "extracted_answer": extracted_answer,
                "is_yes": is_yes,
                "api_provider": provider,
                "response_time": api_metadata.get("response_time", 0),
                "api_status": api_metadata.get("status", "unknown")
            }
            
            return result
            
        except Exception as e:
            # Log error
            self.error_logger.error(f"TRIAL_ERROR|{question_data.get('question_index', 0)}|{model}|{prompt_variation}|{temperature}|{trial_idx}|{str(e)}")
            return None
    
    async def run_experiment_for_question(self, question_data: Dict, question_idx: int):
        """Run full experiment for a single question"""
        logger.info(f"Starting experiment for question {question_idx + 1}: {question_data['question'][:50]}...")
        
        question_results = []
        
        # Get all models to test
        all_models = []
        for provider, models in {**HISTORICAL_MODELS, **NEW_MODELS}.items():
            all_models.extend(models)
        
        # Calculate total trials for this question
        total_trials = len(all_models) * len(PROMPT_VARIATIONS) * len(TEMPERATURES) * TRIALS_PER_COMBINATION
        
        with tqdm(total=total_trials, desc=f"Question {question_idx + 1}") as pbar:
            for model in all_models:
                for prompt_variation in PROMPT_VARIATIONS:
                    for temperature in TEMPERATURES:
                        for trial_idx in range(TRIALS_PER_COMBINATION):
                            result = await self.run_single_trial(
                                question_data, model, prompt_variation, temperature, trial_idx
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
    
    def _create_question_summary(self, results: List[Dict]) -> pd.DataFrame:
        """Create summary statistics for a question"""
        df = pd.DataFrame(results)
        
        summary = df.groupby(['provider', 'model', 'prompt_variation', 'temperature']).agg({
            'is_yes': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        
        summary.columns = ['provider', 'model', 'prompt_variation', 'temperature', 'n_trials', 'n_yes', 'percent_yes', 'std_dev']
        summary['percent_yes'] = (summary['percent_yes'] * 100).round(1)
        summary['std_dev'] = summary['std_dev'].round(3)
        
        return summary
    
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
        """Run the complete experiment"""
        logger.info(f"Starting {'trial' if self.trial_mode else 'full'} experiment")
        logger.info(f"Total questions: {len(self.questions)}")
        
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
        logger.info(f"Experiment completed in {total_time/60:.1f} minutes")
        logger.info(f"Total results collected: {len(self.results)}")
        
        # Generate final data quality report
        self._generate_data_quality_report()
    
    def _generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        try:
            df = pd.DataFrame(self.results)
            
            report = []
            report.append("=" * 80)
            report.append("DATA QUALITY REPORT")
            report.append("=" * 80)
            report.append(f"Total trials: {len(df)}")
            report.append(f"Questions tested: {df['question_index'].nunique()}")
            report.append(f"Models tested: {df['model'].nunique()}")
            report.append(f"Providers: {df['provider'].nunique()}")
            report.append(f"Prompt variations: {df['prompt_variation'].nunique()}")
            report.append(f"Temperatures: {df['temperature'].nunique()}")
            
            # Response distribution
            report.append(f"\nResponse distribution:")
            response_dist = df['normalized_answer'].value_counts()
            for response, count in response_dist.items():
                report.append(f"  {response}: {count} ({count/len(df)*100:.1f}%)")
            
            # Missing data check
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                report.append(f"\nMissing data:")
                for col, missing in missing_data.items():
                    if missing > 0:
                        report.append(f"  {col}: {missing} missing values")
            
            # API status summary
            api_status = df['api_status'].value_counts()
            report.append(f"\nAPI call status:")
            for status, count in api_status.items():
                report.append(f"  {status}: {count}")
            
            # Save report
            report_file = OUTPUT_DIR / "data_quality_report.txt"
            with open(report_file, 'w') as f:
                f.write('\n'.join(report))
            
            logger.info(f"Data quality report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive LLM Yes/No experiments")
    parser.add_argument("--trial", action="store_true", help="Run trial with 5 questions")
    parser.add_argument("--full", action="store_true", help="Run full experiment with 92 questions")
    
    args = parser.parse_args()
    
    # Default to trial mode if no mode specified
    if not args.trial and not args.full:
        args.trial = True
    
    if args.trial and args.full:
        print("Error: Cannot specify both --trial and --full")
        return
    
    # Run experiment
    runner = ComprehensiveExperimentRunner(trial_mode=args.trial)
    await runner.run_full_experiment()


if __name__ == "__main__":
    asyncio.run(main())
