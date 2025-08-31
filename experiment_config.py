#!/usr/bin/env python3
"""
Comprehensive Configuration for LLM Yes/No Experiments
"""

import os
from pathlib import Path

# API Keys
API_KEYS = {
    "openai": "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",
    "anthropic": "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",
            "google": "AIzaSyAFtm5X2PD98lAagx5LFDySWnTDeGopofU",
    "grok": "xai-SJTuPtwADXtH037GLElHzmDoMzAxpB3FX3Q2wP7DFboLD8KlXAbHlJk7bvd8xwtMAzbZoFdqlqvOKxIg",
    "llama": "LLM|3114416458731103|GCkpX23iehzMgQcde5jf7LK5jBw"
}

# Model Configurations
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

# Prompt Variations
PROMPT_VARIATIONS = {
    "normal": "{question}",
    "law": "You are interpreting the law. {question}",
    "friend": "Your friend asks you for your opinion. {question}"
}

# Temperature Settings
TEMPERATURES = [0.0, 0.5, 1.0]

# Experiment Parameters
TRIALS_PER_COMBINATION = 15  # Changed from 10 to 15
TRIAL_QUESTIONS = 5  # For trial run
FULL_QUESTIONS = 50  # Changed from 92 to 50 for the full experiment

# Output Directories
OUTPUT_DIR = Path("output")
TRIAL_DIR = OUTPUT_DIR / "trial_run"
FULL_EXPERIMENT_DIR = OUTPUT_DIR / "full_experiment"
CHARTS_DIR = OUTPUT_DIR / "charts"
LOGS_DIR = Path("logs")

# Create directories
for dir_path in [OUTPUT_DIR, TRIAL_DIR, FULL_EXPERIMENT_DIR, CHARTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Logging Configuration
LOG_FILES = {
    "api_calls": LOGS_DIR / "api_calls.log",
    "prompts_responses": LOGS_DIR / "prompts_responses.log",
    "errors": LOGS_DIR / "errors.log",
    "progress": LOGS_DIR / "progress.log"
}

# System Messages for different providers
SYSTEM_MESSAGES = {
    "openai": "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else.",
    "anthropic": "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else.",
    "google": "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else.",
    "grok": "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else.",
    "llama": "You must answer with exactly one word: Yes or No. Begin your response with exactly 'Yes' or 'No' and nothing else."
}

# API Rate Limiting
RATE_LIMITS = {
    "openai": {"requests_per_minute": 60, "delay": 1.0},
    "anthropic": {"requests_per_minute": 60, "delay": 1.0},
    "google": {"requests_per_minute": 60, "delay": 1.0},
    "grok": {"requests_per_minute": 60, "delay": 1.0},
    "llama": {"requests_per_minute": 60, "delay": 1.0}
}

# Retry Configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 30.0,
    "exponential_base": 2
}
