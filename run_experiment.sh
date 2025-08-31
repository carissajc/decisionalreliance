#!/bin/bash
# Comprehensive LLM Experiment Runner

echo "Comprehensive LLM Yes/No Experiment System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check API keys
echo ""
echo "Checking API keys..."
python -c "
from experiment_config import API_KEYS
for provider, key in API_KEYS.items():
    if key and key != 'your_api_key_here':
        print(f'✓ {provider}: API key configured')
    else:
        print(f'✗ {provider}: API key missing')
"

echo ""
echo "Choose experiment mode:"
echo "1) Trial run (5 questions)"
echo "2) Full experiment (92 questions)"
echo "3) Exit"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Running trial experiment..."
        python comprehensive_experiment_runner.py --trial
        ;;
    2)
        echo "Running full experiment..."
        python comprehensive_experiment_runner.py --full
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
