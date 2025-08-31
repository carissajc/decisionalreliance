# Comprehensive LLM Yes/No Experiment System

This system runs comprehensive experiments across multiple LLM providers, models, prompt variations, and temperature settings.

## Quick Start

### 1. Trial Run (5 questions)
```bash
python comprehensive_experiment_runner.py --trial
```

### 2. Full Experiment (92 questions)
```bash
python comprehensive_experiment_runner.py --full
```

## System Features

- **Real-time output**: Results and charts generated immediately after each question
- **Comprehensive logging**: Detailed logs for API calls, prompts/responses, and errors
- **Robust error handling**: Retry logic and comprehensive error reporting
- **Professional charts**: Three vertical bar charts per question with consistent styling
- **Progressive saving**: Results saved progressively, not waiting for completion

## Output Structure

```
output/
├── trial_run/           # Results from 5-question test
├── full_experiment/     # Results from 92-question run
│   ├── question_1/      # Results and charts for question 1
│   ├── question_2/      # Results and charts for question 2
│   └── ...              # Progressive output as each question completes
├── charts/              # All visualization files (generated progressively)
└── data_quality_report.txt  # Summary of data completeness
```

## Logs

```
logs/
├── api_calls.log        # Detailed API call log
├── prompts_responses.log # Exact prompts and responses
├── errors.log           # Any issues or missing data
└── progress.log         # Progress tracking
```

## Configuration

All settings are in `experiment_config.py`:
- API keys for all providers
- Model lists (historical and new)
- Prompt variations and temperature settings
- Experiment parameters

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## API Keys

The system includes API keys for:
- OpenAI (GPT models)
- Anthropic (Claude models)  
- Google (Gemini models)
- Grok
- Llama

## Chart Types

For each question, the system generates:

1. **Chart 1**: Historical models showing % Yes across all trials
2. **Chart 2**: New models with three prompt variations (Normal, Law, Friend)
3. **Chart 3**: New models with three temperature settings (0.0, 0.5, 1.0)

All charts maintain professional styling with Y-axis from 0-100%.
