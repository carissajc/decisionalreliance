#!/usr/bin/env python3
"""
Test script to test each historical model individually
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# Import our experiment configuration
from experiment_config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalModelTester:
    """Test each historical model individually"""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
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
                self.clients["grok"] = {"api_key": API_KEYS["grok"]}
                logger.info("✓ Grok client initialized (basic)")
            
            # Llama
            if "llama" in API_KEYS:
                self.clients["llama"] = {"api_key": API_KEYS["llama"]}
                logger.info("✓ Llama client initialized (basic)")
                
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
    
    async def _call_openai(self, model: str, prompt: str) -> str:
        """Make OpenAI API call with proper parameter handling"""
        client = self.clients["openai"]
        
        try:
            if model == "gpt-3.5-turbo-instruct":
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    stop=["\n", " ", ".", ","]
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
                        max_completion_tokens=1,
                        temperature=0.0,
                        stop=["\n", " ", ".", ","]
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_MESSAGES["openai"]},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1,
                        temperature=0.0,
                        stop=["\n", " ", ".", ","]
                    )
                return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _call_anthropic(self, model: str, prompt: str) -> str:
        """Make Anthropic Claude API call"""
        client = self.clients["anthropic"]
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": prompt}],
                system=SYSTEM_MESSAGES["anthropic"],
                temperature=0.0
            )
            return response.content[0].text if response.content else ""
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def _call_google(self, model: str, prompt: str) -> str:
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
                "temperature": 0.0
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
    
    async def _call_grok(self, model: str, prompt: str) -> str:
        """Make Grok API call"""
        # Placeholder - implement actual Grok API integration
        return "yes"  # Mock response for testing
    
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
    
    async def test_model(self, provider: str, model: str, prompt: str) -> dict:
        """Test a single model with the given prompt"""
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Testing {provider}/{model}...")
            
            if provider == "openai":
                response = await self._call_openai(model, prompt)
            elif provider == "anthropic":
                response = await self._call_anthropic(model, prompt)
            elif provider == "google":
                response = await self._call_google(model, prompt)
            elif provider == "grok":
                response = await self._call_grok(model, prompt)
            elif provider == "llama":
                response = await self._call_llama(model, prompt, 0.0) # Assuming 0.0 for now
            else:
                raise Exception(f"Unknown provider: {provider}")
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Parse response to extract Yes/No and keep raw text
            raw_text = response.strip() if response else ""
            
            # Extract Yes/No from the beginning of the response
            if raw_text.lower().startswith("yes"):
                extracted_answer = "Yes"
                is_yes = 1
            elif raw_text.lower().startswith("no"):
                extracted_answer = "No"
                is_yes = 0
            else:
                extracted_answer = raw_text  # Keep original if no Yes/No found
                is_yes = None
            
            result = {
                "provider": provider,
                "model": model,
                "status": "success",
                "raw_response": response,
                "extracted_answer": extracted_answer,
                "is_yes": is_yes,
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"✓ {provider}/{model}: {extracted_answer} ({duration:.2f}s)")
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "provider": provider,
                "model": model,
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            }
            
            logger.error(f"✗ {provider}/{model}: {str(e)} ({duration:.2f}s)")
            return result
    
    async def test_all_historical_models(self):
        """Test all historical models with a simple question"""
        test_question = "Is a tomato a vegetable?"
        prompt = f"{test_question}"
        
        logger.info("=" * 80)
        logger.info("TESTING HISTORICAL MODELS")
        logger.info("=" * 80)
        logger.info(f"Test question: {test_question}")
        logger.info("")
        
        results = []
        
        # Test all historical models
        for provider, models in HISTORICAL_MODELS.items():
            logger.info(f"Testing {provider} models...")
            logger.info("-" * 40)
            
            for model in models:
                result = await self.test_model(provider, model, prompt)
                results.append(result)
                
                # Small delay between models
                await asyncio.sleep(0.5)
            
            logger.info("")
        
        # Test all new models for comparison
        logger.info("=" * 80)
        logger.info("TESTING NEW MODELS")
        logger.info("=" * 80)
        
        for provider, models in NEW_MODELS.items():
            logger.info(f"Testing {provider} models...")
            logger.info("-" * 40)
            
            for model in models:
                result = await self.test_model(provider, model, prompt)
                results.append(result)
                
                # Small delay between models
                await asyncio.sleep(0.5)
            
            logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        logger.info(f"Total models tested: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            logger.info("\nSuccessful models:")
            for result in successful:
                logger.info(f"  ✓ {result['provider']}/{result['model']}: {result['extracted_answer']}")
        
        if failed:
            logger.info("\nFailed models:")
            for result in failed:
                logger.info(f"  ✗ {result['provider']}/{result['model']}: {result['error']}")
        
        # Save results to file
        import json
        output_file = Path("test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {output_file}")
        
        return results


async def main():
    """Main entry point"""
    tester = HistoricalModelTester()
    results = await tester.test_all_historical_models()
    return results


if __name__ == "__main__":
    asyncio.run(main())
