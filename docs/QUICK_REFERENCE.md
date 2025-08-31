# QUICK REFERENCE GUIDE

## 🚀 **FAST START COMMANDS**

### **Run Fast Experiment (Recommended)**
```bash
python run_fast_experiment.py
```
- **Speed**: 20-30x faster than sequential
- **Output**: `output/trials_fast_parallel.csv`
- **Time**: ~5-10 minutes for full experiment

### **Analyze Temperature Effects**
```bash
python analyze_temperature_patterns.py
```
- **Output**: 5 detailed tables + visualization
- **File**: `output/temperature_effects_analysis.png`

### **Test Specific Provider**
```bash
# OpenAI only
python run_simple_experiment.py

# Claude models
python test_claude.py

# Gemini models  
python test_gemini.py

# Mistral models
python test_mistral.py

# Grok models
python test_grok.py
```

---

## 📊 **KEY FINDINGS SUMMARY**

### **Temperature Effects (Minimal)**
- **Overall**: Only 0.5 percentage points difference
- **Q_ONLY**: 1.3 points sensitivity
- **Q_SUBSET_SET_ORDINARY**: 2.9 points sensitivity  
- **Q_SUBSET_SET_STATUTE_ORDINARY**: 3.0 points sensitivity

### **Context Matters More Than Temperature**
- **Q_ONLY**: 20% yes rate
- **Q_SUBSET_SET_ORDINARY**: 84-87% yes rate
- **Q_SUBSET_SET_STATUTE_ORDINARY**: 66-70% yes rate

### **Provider Performance**
- **OpenAI/Simple**: Most consistent (72-74% yes rate)
- **Claude**: Moderate consistency (48-50% yes rate)
- **Mistral**: Most variable (22-29% yes rate)
- **Grok**: Very consistent (33% yes rate across all temps)

---

## 🔧 **COMMON TASKS**

### **1. Run New Experiment**
```bash
# Fast parallel (recommended)
python run_fast_experiment.py

# Full control with options
python run_yesno_panel.py --replicates 50 --concurrency 20
```

### **2. Analyze Results**
```bash
# Temperature effects
python analyze_temperature_patterns.py

# All results summary
python analyze_all_results.py

# Basic analysis
python analyze_results.py
```

### **3. Test Specific Models**
```bash
# Edit model list in script header
# Run provider-specific test
python test_[provider].py
```

### **4. Check API Keys**
```bash
# Test OpenAI
python -c "from openai import OpenAI; print('✓ OpenAI OK')"

# Test Claude
python -c "import anthropic; print('✓ Claude OK')"

# Test Gemini
python -c "import google.generativeai; print('✓ Gemini OK')"
```

---

## 📁 **FILE LOCATIONS**

### **Input Data**
- **Question Data**: `hypotheticals.xlsx`
- **Configuration**: `requirements.txt`

### **Output Data**
- **Complete Results**: `output/trials_all_providers_updated.csv`
- **Fast Experiment**: `output/trials_fast_parallel.csv`
- **Provider Results**: `output/trials_[provider].csv`

### **Visualizations**
- **Temperature Analysis**: `output/temperature_effects_analysis.png`
- **Model Charts**: `output/yes_rate_by_model__[variant]__[temp].png`
- **Heatmaps**: `output/heatmap__[variant]__[temp].png`

---

## ⚡ **PERFORMANCE TIPS**

### **Speed Optimization**
```python
# In run_fast_experiment.py
max_concurrent = 30  # Increase for faster execution
replicates = 5       # Decrease for faster testing
```

### **Memory Management**
```python
# In run_yesno_panel.py
--replicates 10      # Reduce for memory issues
--concurrency 8      # Reduce for stability
```

### **API Limits**
```python
# Reduce concurrency if hitting rate limits
max_concurrent = 10  # Conservative setting
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Errors**

#### **"Model not found"**
```bash
# Check model list in script
# Remove unsupported models
# Update to latest model names
```

#### **"API rate limit exceeded"**
```bash
# Reduce concurrency
# Wait between runs
# Check API quota status
```

#### **"Memory error"**
```bash
# Reduce replicates
# Process in smaller batches
# Close other applications
```

#### **"Visualization failed"**
```bash
# Install dependencies
pip install matplotlib seaborn
# Check display settings
```

---

## 📈 **EXPERIMENT DESIGN**

### **Question Variants**
1. **Q_ONLY**: Basic question
2. **Q_SUBSET_SET_ORDINARY**: + context
3. **Q_SUBSET_SET_STATUTE_ORDINARY**: + context + statute

### **Temperature Settings**
- **0.0**: Deterministic
- **0.2**: Low randomness  
- **0.5**: Moderate randomness

### **Replication Strategy**
- **Default**: 5 replicates
- **Production**: 100+ replicates
- **Testing**: 1-2 replicates

---

## 🔑 **API KEYS REQUIRED**

```bash
# Required for experiments
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export MISTRAL_API_KEY="..."

# Optional (hardcoded in scripts)
# GROK_API_KEY in test_grok.py
```

---

## 📞 **GETTING HELP**

### **Documentation Files**
- **Overview**: `WORKSPACE_ORGANIZATION.md`
- **Script Details**: `SCRIPT_DOCUMENTATION.md`
- **Project Info**: `README.md`

### **Quick Checks**
```bash
# Check script status
python -c "import sys; print('Python:', sys.version)"

# Check dependencies
pip list | grep -E "(openai|anthropic|google|mistral)"

# Check output directory
ls -la output/
```

---

*Last Updated: 2025-08-12*  
*Quick Reference Status: Complete* 