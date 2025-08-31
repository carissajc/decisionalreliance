# SCRIPT DOCUMENTATION

## 📋 **COMPREHENSIVE SCRIPT REFERENCE**

---

## 🚀 **CORE EXPERIMENT SCRIPTS**

### **1. `run_fast_experiment.py` - HIGH-SPEED PARALLEL EXPERIMENT**

#### **Purpose**
Fastest experiment runner using async parallel execution for 20-30x speed improvement.

#### **Key Features**
- **Concurrency**: Up to 30 simultaneous API calls
- **All Providers**: OpenAI, Claude, Gemini, Mistral, Grok
- **Smart Batching**: Creates all tasks upfront, executes in parallel
- **Error Isolation**: Failed calls don't stop others
- **Progress Tracking**: Real-time speed metrics

#### **Usage**
```bash
python run_fast_experiment.py
```

#### **Configuration**
```python
# In script header
max_concurrent = 30  # Adjust based on API limits
replicates = 5       # Per combination
```

#### **Expected Output**
```
================================================================================
FAST PARALLEL EXPERIMENT - ALL API PROVIDERS
================================================================================
Question: Is a tomato a vegetable?
Max concurrent requests: 30
Replicates per combination: 5
Total trials to run: 1,305

Starting 1,305 parallel trials...
Completed 1,200 trials in 45.2 seconds
Average speed: 26.5 trials/second
```

#### **Output Files**
- `output/trials_fast_parallel.csv` - Complete results
- Console summary with provider performance

---

### **2. `run_yesno_panel.py` - FULL EXPERIMENTAL CONTROL**

#### **Purpose**
Comprehensive experiment runner with full experimental design capabilities.

#### **Key Features**
- **Full Control**: All parameters configurable
- **Robust Error Handling**: Retries, exponential backoff
- **Incremental Saving**: Results saved as they complete
- **Advanced Logging**: Detailed execution logs
- **Flexible Input**: Excel file reading, multiple questions

#### **Usage**
```bash
# Basic run
python run_yesno_panel.py

# With options
python run_yesno_panel.py --replicates 100 --concurrency 16 --use-seed
```

#### **CLI Options**
```bash
--replicates N           # Number of replicates (default: 100)
--concurrency N          # Max concurrent requests (default: 8)
--use-seed               # Use deterministic seeds per replicate
--cache-buster           # Append UUID to prevent caching
--include-case           # Include case text in prompts
--truncate-statute-chars N  # Limit statute length (default: 4000)
```

#### **Expected Output**
```
Running experiment with 100 replicates per combination...
Progress: [████████████████████] 100%
Completed 1,305 trials in 12.3 minutes
Saved results to output/trials.csv
```

---

### **3. `run_simple_experiment.py` - OPENAI-ONLY TESTING**

#### **Purpose**
Simplified, tested experiment runner for OpenAI models only.

#### **Key Features**
- **OpenAI Focus**: GPT-3.5, GPT-4, GPT-4o models
- **Simple Design**: Basic experiment loop, easy to debug
- **Tested**: Known working configuration
- **Fast**: Quick execution for testing

#### **Usage**
```bash
python run_simple_experiment.py
```

#### **Models Tested**
- gpt-3.5-turbo-0125
- gpt-3.5-turbo-1106
- gpt-3.5-turbo-instruct
- gpt-4-turbo-2024-04-09
- gpt-4-0125-preview
- gpt-4-0613
- gpt-4o variants

---

## 🔬 **PROVIDER-SPECIFIC TEST SCRIPTS**

### **4. `test_claude.py` - CLAUDE MODEL TESTING**

#### **Purpose**
Dedicated Claude experiment runner with proper API handling.

#### **Models Tested**
- claude-3-5-sonnet-20241022
- claude-3-5-sonnet-latest
- claude-3-7-sonnet-20250219
- claude-3-7-sonnet-latest
- claude-opus-4-20250514
- claude-sonnet-4-20250514

#### **Usage**
```bash
python test_claude.py
```

#### **API Requirements**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

---

### **5. `test_gemini.py` - GEMINI MODEL TESTING**

#### **Purpose**
Google Gemini experiment runner with generation config.

#### **Models Tested**
- gemini-1.5-pro-latest
- gemini-2.0-flash
- gemini-2.5-flash
- gemini-2.5-flash-preview-05-20

#### **Usage**
```bash
python test_gemini.py
```

#### **API Requirements**
```bash
export GOOGLE_API_KEY="your-key-here"
```

---

### **6. `test_mistral.py` - MISTRAL MODEL TESTING**

#### **Purpose**
Mistral AI experiment runner.

#### **Models Tested**
- mistral-large-latest
- mistral-medium-latest
- mistral-small-latest

#### **Usage**
```bash
python test_mistral.py
```

#### **API Requirements**
```bash
export MISTRAL_API_KEY="your-key-here"
```

---

### **7. `test_grok.py` - GROK MODEL TESTING**

#### **Purpose**
xAI Grok experiment runner using REST API.

#### **Models Tested**
- grok-3-latest
- grok-3-beta
- grok-4-0709

#### **Usage**
```bash
python test_grok.py
```

#### **API Requirements**
```bash
# API key hardcoded in script for testing
```

---

## 📊 **ANALYSIS & VISUALIZATION SCRIPTS**

### **8. `analyze_temperature_patterns.py` - TEMPERATURE EFFECTS**

#### **Purpose**
Comprehensive temperature pattern analysis with improved visualizations.

#### **Features**
- **5 Separate Tables**: Clear organization of results
- **Bar Charts Only**: No heatmaps, clean visualizations
- **Sensitivity Analysis**: Temperature effect quantification
- **Provider Comparison**: Cross-provider temperature effects

#### **Usage**
```bash
python analyze_temperature_patterns.py
```

#### **Output Tables**
1. **Overall Response Patterns by Temperature**
2. **Response Patterns by Temperature and Variant**
3. **Response Patterns by Temperature and Provider**
4. **Detailed Breakdown by Temperature, Variant, and Provider**
5. **Temperature Effect Analysis by Variant**

#### **Visualizations**
- `output/temperature_effects_analysis.png` - 4-panel chart
- Overall patterns, variant effects, provider effects, sensitivity

---

### **9. `analyze_all_results.py` - COMPREHENSIVE ANALYSIS**

#### **Purpose**
Cross-provider results analysis and model ranking.

#### **Features**
- **Provider Comparison**: Performance across all providers
- **Model Rankings**: Top performers by yes rate
- **Variant Analysis**: Response patterns by question type
- **Statistical Summary**: Counts, means, distributions

#### **Usage**
```bash
python analyze_all_results.py
```

#### **Output**
- Console summary with key findings
- Provider performance tables
- Model ranking tables

---

### **10. `analyze_results.py` - BASIC ANALYSIS**

#### **Purpose**
Simple results analysis and visualization.

#### **Features**
- **Basic Charts**: Bar charts and provider comparisons
- **Simple Statistics**: Counts and percentages
- **Quick Overview**: Fast results summary

#### **Usage**
```bash
python analyze_results.py
```

---

## 📁 **DATA FILE STRUCTURE**

### **Input Data Format**
```excel
# hypotheticals.xlsx
question                    | statute                    | subset              | set
Is a tomato a vegetable?   | In botanical terms...      | Botanical class.    | Fruits vs Vegetables
```

### **Output Data Format**
```csv
# trials_[provider].csv
timestamp_utc,question_index,model,variant,temperature,replicate_idx,raw_text,normalized_answer,is_yes,api_provider
2025-08-12T21:57:04.801Z,1,gpt-4,Q_ONLY,0.0,0,yes,yes,1,openai
```

### **Aggregated Data Format**
```csv
# aggregates_all_providers.csv
question_index,model,variant,temperature,count,yes_count,yes_rate
1,gpt-4,Q_ONLY,0.0,5,1,0.2
```

---

## ⚙️ **CONFIGURATION & SETUP**

### **Environment Variables**
```bash
# Required for all experiments
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export MISTRAL_API_KEY="..."

# Optional for specific providers
export GROK_API_KEY="xai-..."  # Hardcoded in test_grok.py
```

### **Dependencies Installation**
```bash
pip install -r requirements.txt
```

### **API Key Validation**
```bash
# Test OpenAI
python -c "from openai import OpenAI; print('OpenAI OK')"

# Test Claude
python -c "import anthropic; print('Claude OK')"

# Test Gemini
python -c "import google.generativeai; print('Gemini OK')"
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

#### **API Rate Limits**
```bash
# Reduce concurrency
python run_fast_experiment.py  # Edit max_concurrent=10
```

#### **Model Not Found**
```bash
# Check model list in script
# Remove unsupported models
```

#### **Memory Issues**
```bash
# Reduce replicates
python run_yesno_panel.py --replicates 10
```

#### **Visualization Errors**
```bash
# Install matplotlib dependencies
pip install matplotlib seaborn
```

---

## 📈 **PERFORMANCE METRICS**

### **Speed Comparison**
- **Sequential**: ~2-3 hours for 1,305 trials
- **Parallel (30 concurrent)**: ~5-10 minutes
- **Speed Improvement**: 20-30x faster

### **Resource Usage**
- **Memory**: ~500MB for full dataset
- **CPU**: Minimal (I/O bound)
- **Network**: Depends on API response times

---

*Last Updated: 2025-08-12*  
*Total Scripts: 10*  
*Documentation Status: Complete* 