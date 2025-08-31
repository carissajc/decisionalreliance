# 🎯 MASTER INDEX - DECISIONAL RELIANCE WORKSPACE

## 📚 **DOCUMENTATION NAVIGATION**

### **🚀 QUICK START**
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Fast commands and common tasks
- **[README.md](README.md)** - Project overview and installation

### **📋 COMPREHENSIVE GUIDES**
- **[WORKSPACE_ORGANIZATION.md](WORKSPACE_ORGANIZATION.md)** - Complete file organization
- **[SCRIPT_DOCUMENTATION.md](SCRIPT_DOCUMENTATION.md)** - Detailed script reference
- **[docs/WORKSPACE_SUMMARY.md](docs/WORKSPACE_SUMMARY.md)** - Current status and findings

### **🔧 ORGANIZED DOCUMENTATION**
- **[docs/scripts/SCRIPT_ORGANIZATION.md](docs/scripts/SCRIPT_ORGANIZATION.md)** - Script categorization
- **[docs/data/DATA_ORGANIZATION.md](docs/data/DATA_ORGANIZATION.md)** - Data file organization
- **[docs/visualizations/VISUALIZATION_ORGANIZATION.md](docs/visualizations/VISUALIZATION_ORGANIZATION.md)** - Chart organization

---

## 🎯 **WHAT YOU NEED TO KNOW**

### **🏆 KEY ACHIEVEMENTS**
- ✅ **6,176 trials completed** across 6 API providers
- ✅ **29 models tested** with 3 variants and 3 temperatures
- ✅ **20-30x speed improvement** with parallel execution
- ✅ **Comprehensive analysis** of temperature effects
- ✅ **Production-ready scripts** for all major LLM providers

### **🔍 KEY FINDINGS**
- **Temperature has minimal effect**: Only 0.5 percentage points overall
- **Context matters more**: Q_SUBSET_SET_ORDINARY shows 84-87% yes rate vs 20% for Q_ONLY
- **Provider differences**: OpenAI most consistent, Mistral most variable
- **Model consistency**: Most models stable across temperatures

---

## 🚀 **RECOMMENDED WORKFLOW**

### **1. NEW EXPERIMENT**
```bash
# Fast parallel execution (recommended)
python run_fast_experiment.py

# Full experimental control
python run_yesno_panel.py --replicates 100 --concurrency 20
```

### **2. ANALYZE RESULTS**
```bash
# Temperature effects analysis
python analyze_temperature_patterns.py

# Comprehensive results
python analyze_all_results.py
```

### **3. TEST SPECIFIC PROVIDERS**
```bash
# OpenAI only
python run_simple_experiment.py

# Individual providers
python test_claude.py
python test_gemini.py
python test_mistral.py
python test_grok.py
```

---

## 📁 **FILE ORGANIZATION**

### **📝 SCRIPTS (25 files)**
- **Core Experiments**: `run_yesno_panel.py`, `run_fast_experiment.py`, `run_simple_experiment.py`
- **Provider Tests**: `test_claude.py`, `test_gemini.py`, `test_mistral.py`, `test_grok.py`
- **Analysis**: `analyze_temperature_patterns.py`, `analyze_all_results.py`, `analyze_results.py`
- **Organization**: `organize_workspace.py`

### **📊 DATA FILES**
- **Input**: `hypotheticals.xlsx` (tomato question)
- **Output**: `output/trials_all_providers_updated.csv` (6,176 trials)
- **Aggregates**: `output/aggregates_all_providers.csv`

### **📈 VISUALIZATIONS**
- **Main**: `output/temperature_effects_analysis.png` (4-panel analysis)
- **Model Charts**: 27 performance charts by variant/temperature
- **Legacy**: Archived heatmaps in `docs/archive/`

---

## 🎨 **VISUALIZATION IMPROVEMENTS**

### **✅ COMPLETED**
- ❌ **Removed heatmaps** as requested
- ✅ **Bar charts only** with distinct colors
- ✅ **More spacious layout** (20x16 figure size)
- ✅ **Value labels** on each bar
- ✅ **Separate tables** instead of combined output
- ✅ **Temperature sensitivity analysis** with quantification

### **📊 CHART TYPES**
1. **Overall Yes Rate by Temperature** - Bar chart
2. **Yes Rate by Temperature and Variant** - Grouped bar chart
3. **Yes Rate by Temperature and Provider** - Grouped bar chart
4. **Temperature Sensitivity by Variant** - Bar chart

---

## 🔑 **API PROVIDERS & MODELS**

### **🤖 OPENAI (9 models)**
- GPT-3.5 variants, GPT-4 variants, GPT-4o variants
- **Status**: ✅ Fully tested and working

### **🧠 CLAUDE (9 models)**
- Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude Opus 4, Claude Sonnet 4
- **Status**: ✅ Fully tested and working

### **🔮 GEMINI (8 models)**
- Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.5 Flash
- **Status**: ✅ Fully tested and working

### **🌪️ MISTRAL (3 models)**
- Mistral Large, Medium, Small
- **Status**: ✅ Fully tested and working

### **🚀 GROK (3 models)**
- Grok 3 variants, Grok 4
- **Status**: ✅ Fully tested and working

---

## 📊 **EXPERIMENTAL DESIGN**

### **❓ QUESTION VARIANTS**
1. **Q_ONLY**: "Is a tomato a vegetable?"
2. **Q_SUBSET_SET_ORDINARY**: + botanical classification context
3. **Q_SUBSET_SET_STATUTE_ORDINARY**: + botanical context + statute

### **🌡️ TEMPERATURE SETTINGS**
- **0.0**: Deterministic responses
- **0.2**: Low randomness
- **0.5**: Moderate randomness

### **🔄 REPLICATION STRATEGY**
- **Default**: 5 replicates per combination
- **Production**: 100+ replicates
- **Testing**: 1-2 replicates

---

## 🚨 **TROUBLESHOOTING**

### **⚡ SPEED ISSUES**
- Use `run_fast_experiment.py` for 20-30x speed improvement
- Increase `max_concurrent` parameter (default: 30)
- Reduce replicates for faster testing

### **🔌 API ISSUES**
- Check API keys in environment variables
- Reduce concurrency if hitting rate limits
- Use provider-specific test scripts for debugging

### **💾 MEMORY ISSUES**
- Reduce replicates parameter
- Process in smaller batches
- Close other applications

---

## 📈 **PERFORMANCE METRICS**

### **⏱️ SPEED COMPARISON**
- **Sequential execution**: ~2-3 hours for 1,305 trials
- **Parallel execution (30 concurrent)**: ~5-10 minutes
- **Speed improvement**: **20-30x faster**

### **💻 RESOURCE USAGE**
- **Memory**: ~500MB for full dataset
- **CPU**: Minimal (I/O bound)
- **Network**: Depends on API response times

---

## 🎯 **NEXT STEPS RECOMMENDATIONS**

### **🔬 RESEARCH OPPORTUNITIES**
1. **Expand question set** beyond tomato question
2. **Test more temperature values** (0.1, 0.3, 0.7, 1.0)
3. **Add more context variants** for comprehensive analysis
4. **Cross-validate findings** with different question types

### **🚀 TECHNICAL IMPROVEMENTS**
1. **Add more LLM providers** (Cohere, AI21, etc.)
2. **Implement result caching** for cost optimization
3. **Add statistical significance testing**
4. **Create interactive dashboards**

### **📊 ANALYSIS ENHANCEMENTS**
1. **Response time analysis** by model/provider
2. **Cost analysis** per trial and provider
3. **Model consistency metrics** across temperatures
4. **Prompt engineering optimization**

---

## 📞 **GETTING HELP**

### **📚 DOCUMENTATION ORDER**
1. **Start with**: `QUICK_REFERENCE.md`
2. **For details**: `SCRIPT_DOCUMENTATION.md`
3. **For organization**: `WORKSPACE_ORGANIZATION.md`
4. **For current status**: `docs/WORKSPACE_SUMMARY.md`

### **🔧 QUICK CHECKS**
```bash
# Check Python and dependencies
python -c "import sys; print('Python:', sys.version)"
pip list | grep -E "(openai|anthropic|google|mistral)"

# Check output directory
ls -la output/

# Run organization script
python organize_workspace.py
```

---

## 🏆 **WORKSPACE STATUS**

- **📊 Data Collection**: ✅ **COMPLETE** (6,176 trials)
- **🔬 Analysis**: ✅ **COMPLETE** (temperature effects, provider comparison)
- **📈 Visualization**: ✅ **COMPLETE** (improved bar charts, separate tables)
- **📚 Documentation**: ✅ **COMPLETE** (comprehensive guides)
- **🗂️ Organization**: ✅ **COMPLETE** (structured file system)
- **🚀 Performance**: ✅ **OPTIMIZED** (20-30x speed improvement)

**Overall Status**: 🎉 **PRODUCTION READY**

---

*Last Updated: 2025-08-12*  
*Total Documentation Files: 8*  
*Workspace Status: Fully Organized & Documented* 