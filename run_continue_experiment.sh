#!/bin/bash

# Continue Full LLM Experiment Script
# This script continues the experiment from where it left off

echo "=== Continuing Full LLM Experiment ==="
echo "Starting at: $(date)"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing/upgrading dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if hypotheticals.xlsx exists
if [ ! -f "hypotheticals.xlsx" ]; then
    echo "ERROR: hypotheticals.xlsx not found!"
    echo "Please ensure the file is in the current directory."
    exit 1
fi

# Check if experiment_config.py exists
if [ ! -f "experiment_config.py" ]; then
    echo "ERROR: experiment_config.py not found!"
    echo "Please ensure the configuration file exists."
    exit 1
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p output/full_experiment
mkdir -p output/charts
mkdir -p logs

# Run the continued experiment
echo "Starting continued experiment..."
echo "This will process all remaining questions with both historical and new models."
echo

python continue_full_experiment.py

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "=== Experiment completed successfully! ==="
    echo "Completed at: $(date)"
    echo
    echo "Results saved in:"
    echo "  - output/full_experiment/ (individual question results)"
    echo "  - output/charts/ (generated charts)"
    echo "  - output/combined_summary_all_questions.csv (combined results)"
    echo "  - output/data_quality_report_final.txt (data quality report)"
    echo "  - logs/continue_experiment.log (detailed logs)"
else
    echo
    echo "=== Experiment failed! ==="
    echo "Check the logs for details: logs/continue_experiment.log"
    exit 1
fi
