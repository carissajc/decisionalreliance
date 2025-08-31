# Cursor Prompt: Comprehensive LLM Yes/No Experiment System

## Overview
I need you to implement a comprehensive LLM experiment system that tests multiple models across different prompt variations and temperature settings. The system should show raw responses exactly as returned by each model, with no normalization or consistency enforcement.

## Core Requirements

### 1. **Raw Response Preservation**
- **NEVER normalize or modify responses** - show exactly what each model returns
- **Preserve response inconsistencies** - if one model says "yes" and another says "no", show both exactly as returned
- **No response cleaning** - keep original capitalization, punctuation, spacing
- **Display raw text** in all outputs, charts, and logs

### 2. **Model Configuration**
- **Historical Models** (16 total):
  - **OpenAI**: gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-instruct, gpt-4-turbo-2024-04-09, gpt-4-0125-preview, gpt-4-0613, gpt-4o-2024-11-20, gpt-4o-2024-08-06, gpt-4o-2024-05-13
  - **Anthropic**: claude-3-5-sonnet-20241022, claude-3-5-sonnet-20240620, claude-3-5-sonnet-latest, claude-3-5-haiku-20241022, claude-3-5-haiku-latest, claude-3-7-sonnet-20250219, claude-3-7-sonnet-latest
- **New Models** (5 total):
  - **OpenAI**: gpt-4.1-2025-04-14
  - **Anthropic**: claude-sonnet-4-20250514
  - **Grok**: grok-4-0709
  - **Google**: gemini-2.5-flash
  - **Llama**: Llama-4-Maverick-17B-128E-Instruct-FP8

### 3. **Experiment Parameters**
- **Prompt Variations**: 
  - Normal: "{question}"
  - Law: "You are interpreting the law. {question}"
  - Friend: "Your friend asks you for your opinion. {question}"
- **Temperatures**: 0.0, 0.5, 1.0
- **Trials per combination**: 10
- **Questions**: Load from `hypotheticals.xlsx` (92 total questions)
- **max_tokens**: 3 (for all models)

### 4. **API Integration Requirements**
- **OpenAI**: Use `max_tokens=3` for older models, `max_completion_tokens=3` for GPT-4.1+
- **Anthropic**: Use `max_tokens=3`
- **Google Gemini**: Use `max_output_tokens=3` with retry logic for 500 errors
- **Grok**: Use `xai-sdk` with proper error handling
- **Llama**: Use `aiohttp` for REST API calls

### 5. **Output Requirements**
- **Three charts per question** (one figure with three subplots):
  - **Chart 1 (Left)**: Historical models showing % Yes across all trials
  - **Chart 2 (Middle)**: New models grouped by model with three bars for prompt variations
  - **Chart 3 (Right)**: Same new models grouped by model with three bars for temperature settings
- **Y-axis**: Always 0-100% for "% Yes"
- **Raw response logging**: Exact prompt and response for every API call
- **Failed trial tracking**: Include failed trials in results with error details

### 6. **Data Structure**
- **Results CSV**: Include `raw_answer` field (exact model response)
- **No normalization**: Keep original response text
- **Error tracking**: Log all API failures with detailed error messages
- **Progress monitoring**: Real-time updates and estimated completion times

### 7. **System Messages**
```
"You must answer strictly with exactly one word in lowercase: yes or no."
```
- Apply to all models consistently
- Don't enforce response format - just request it

### 8. **Error Handling & Robustness**
- **Retry logic**: 3 attempts with exponential backoff for server errors
- **Failed trial inclusion**: Count and report failed trials, don't ignore them
- **API error logging**: Detailed error messages with status codes
- **Data quality reporting**: Summary of successful vs failed trials

### 9. **File Organization**
```
decisionalreliance/
├── experiment_config.py          # Central configuration
├── comprehensive_experiment_runner.py  # Main experiment logic
├── chart_generator.py            # Visualization
├── data/
│   └── hypotheticals.xlsx        # 92 questions
├── output/
│   ├── trial_run/                # 5 question test results
│   ├── full_experiment/          # 92 question results
│   └── charts/                   # Generated visualizations
└── logs/                         # Detailed logging
```

### 10. **Key Implementation Points**

#### **Response Handling**
```python
def _normalize_response(self, text: str) -> Tuple[str, Optional[int]]:
    """Return raw response text without normalization"""
    # Don't normalize - return exactly what the model returned
    raw_text = text.strip() if text else ""
    
    # For backward compatibility, still calculate is_yes if possible
    # but don't change the raw text
    if raw_text.lower() == "yes":
        is_yes = 1
    elif raw_text.lower() == "no":
        is_yes = 0
    else:
        is_yes = None
        
    return raw_text, is_yes
```

#### **API Call Structure**
```python
# OpenAI with proper parameter handling
if "gpt-4.1" in model:
    response = client.chat.completions.create(
        model=model,
        messages=[...],
        max_completion_tokens=3,  # For newer models
        temperature=temperature
    )
else:
    response = client.chat.completions.create(
        model=model,
        messages=[...],
        max_tokens=3,  # For older models
        temperature=temperature
    )
```

#### **Failed Trial Tracking**
```python
# Include failed trials in results
if result:
    question_results.append(result)
    self.results.append(result)
else:
    # Log failed trial for debugging
    failed_trial = {
        "model": model,
        "prompt_variation": prompt_variation,
        "temperature": temperature,
        "trial_idx": trial_idx,
        "status": "failed",
        "error": str(e)
    }
    # Add to results or separate failed trials log
```

### 11. **Testing Requirements**
- **Trial run first**: Test with 5 questions before full experiment
- **Model availability testing**: Verify each model works before running experiments
- **API key validation**: Check all API keys are working
- **Error simulation**: Test error handling with invalid parameters

### 12. **Output Examples**
- **Raw Response**: "No" (not "no")
- **Mixed Responses**: "Yes", "No", "no", "YES" (preserve exactly as returned)
- **Empty Responses**: "" (show empty string, don't hide it)
- **Error Responses**: Include error details in results

## Critical Success Factors

1. **Raw Data Preservation**: Never modify or normalize model responses
2. **Complete Error Tracking**: Include failed trials in all outputs
3. **Real API Integration**: No mock responses for Grok/Llama
4. **Parameter Compatibility**: Use correct token parameters for each model
5. **Comprehensive Logging**: Log every API call, success, and failure
6. **Progress Monitoring**: Real-time updates during long experiments
7. **Data Quality**: Ensure all 21 models are tested and results collected

## Expected Outcomes

- **21 working models** (16 historical + 5 new)
- **Raw response preservation** in all outputs
- **Complete error tracking** for failed trials
- **Professional visualizations** with consistent styling
- **Comprehensive logging** for debugging and analysis
- **Robust error handling** for production use

## Implementation Priority

1. **Fix API integration** (remove mocks, implement real calls)
2. **Remove normalization** (preserve raw responses)
3. **Implement error tracking** (include failed trials)
4. **Update parameter handling** (max_tokens=3, proper model-specific params)
5. **Test with trial run** (5 questions, verify all models working)
6. **Run full experiment** (92 questions, all models, all variations)

Remember: **The goal is to show exactly what each model returns, not to enforce consistency or normalize responses.**

