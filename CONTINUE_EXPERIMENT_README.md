# Continue Full LLM Experiment

## Overview
This directory contains the scripts and configuration needed to continue the LLM experiment from where it left off, incorporating all existing data and running both historical and new models for the remaining questions.

## Current Status
- **Questions 1-26**: ✅ COMPLETED
  - Historical models: 5 trials per temperature (0.0, 0.5, 1.0) - no prompt variations
  - New models: 15 trials per prompt variation × temperature combination
- **Questions 27-92**: 🔄 PENDING (66 questions remaining)

## What This Script Does

### 1. **Loads Existing Data**
- Automatically detects completed questions
- Incorporates all existing trial results
- No data loss or duplication

### 2. **Runs Both Model Types**
- **Historical Models**: 16 models × 3 temperatures × 15 trials = 720 trials per question
- **New Models**: 5 models × 3 prompt variations × 3 temperatures × 15 trials = 675 trials per question
- **Total**: 1,395 trials per question

### 3. **Full Experimental Protocol**
- **Prompt Variations**: Normal, Law, Friend
- **Temperature Settings**: 0.0, 0.5, 1.0
- **Trial Count**: 15 trials per combination (increased from 5 for historical models)

### 4. **Real-Time Output**
- Generates charts immediately after each question
- Saves results progressively
- Detailed logging and progress tracking

## Files Created

### Scripts
- `continue_full_experiment.py` - Main experiment runner
- `run_continue_experiment.sh` - Shell script to run the experiment

### Configuration
- `experiment_config.py` - All API keys, model lists, and settings

## How to Run

### Option 1: Using the Shell Script (Recommended)
```bash
chmod +x run_continue_experiment.sh
./run_continue_experiment.sh
```

### Option 2: Direct Python Execution
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python continue_full_experiment.py
```

## Expected Output

### For Each Question (27-92):
1. **Individual Results Directory**: `output/full_experiment/question_X/`
   - `trials_question_X.csv` - Raw trial data
   - `summary_question_X.csv` - Summary statistics
   - `failed_trials_question_X.csv` - Any failed trials

2. **Charts**: `output/charts/question_X_charts.png`
   - Chart 1: Historical models (temperature variations)
   - Chart 2: New models (prompt variations)
   - Chart 3: New models (temperature effects)

### Final Outputs:
- `output/combined_summary_all_questions.csv` - All results combined
- `output/data_quality_report_final.txt` - Comprehensive data quality report
- `logs/continue_experiment.log` - Detailed execution logs

## Model Coverage

### Historical Models (Chart 1)
- **OpenAI**: gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-instruct, gpt-4-turbo-2024-04-09, gpt-4-0125-preview, gpt-4-0613, gpt-4o-2024-11-20, gpt-4o-2024-08-06, gpt-4o-2024-05-13
- **Anthropic**: claude-3-5-sonnet-20241022, claude-3-5-sonnet-20240620, claude-3-5-sonnet-latest, claude-3-5-haiku-20241022, claude-3-5-haiku-latest, claude-3-7-sonnet-20250219, claude-3-7-sonnet-latest

### New Models (Charts 2 & 3)
- **OpenAI**: gpt-4.1-2025-04-14
- **Anthropic**: claude-sonnet-4-20250514
- **Grok**: grok-4-0709
- **Google**: gemini-2.0-flash
- **Llama**: Llama-4-Maverick-17B-128E-Instruct-FP8

## Data Integrity Features

### Robustness Checks
- **Trial Count Validation**: Ensures exactly 15 trials per combination
- **Response Validation**: Flags unexpected responses
- **Missing Data Reporting**: Documents any failed API calls
- **Progress Monitoring**: Real-time tracking with estimated completion times

### Error Handling
- **API Retry Logic**: Exponential backoff for transient failures
- **Graceful Degradation**: Continues processing even if some trials fail
- **Detailed Logging**: Comprehensive error reporting and debugging information

## Estimated Runtime
- **Per Question**: ~15-20 minutes (1,395 API calls)
- **Total Remaining**: ~16-22 hours for 66 questions
- **Progress Updates**: Real-time status for each question

## Monitoring Progress

### During Execution
- Watch the console output for real-time progress
- Check `logs/continue_experiment.log` for detailed information
- Monitor `output/full_experiment/` for new question directories

### After Completion
- Review `output/data_quality_report_final.txt` for data completeness
- Check `output/combined_summary_all_questions.csv` for all results
- Examine individual question charts in `output/charts/`

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Script includes delays between calls
2. **Network Errors**: Automatic retry logic with exponential backoff
3. **Memory Issues**: Progressive saving prevents memory buildup

### If Script Fails
1. Check `logs/continue_experiment.log` for error details
2. Verify API keys in `experiment_config.py`
3. Ensure all dependencies are installed
4. Restart the script - it will resume from where it left off

## Next Steps
After running this script:
1. All 92 questions will have complete data
2. Three comprehensive charts per question
3. Combined summary of all results
4. Data quality validation report
5. Ready for final analysis and visualization

## Support
If you encounter any issues:
1. Check the logs for detailed error information
2. Verify your API keys and quotas
3. Ensure stable internet connection
4. Monitor system resources (memory, disk space)
