# Temperature 0.3 Tests

This folder contains new versions of the original tests with temperature set to 0.3 instead of 0.0.

## Overview

These tests are identical to the original tests in every way except for the temperature setting:
- **Original tests**: Temperature = 0.0 (deterministic)
- **New tests**: Temperature = 0.3 (slightly more variable)

## Tests Included

### 1. GPT-4.1 Prompt Variations Test (Temperature 0.3)
- **File**: `gpt4_1_prompt_test_temp_0.3.py`
- **Question**: "Does transporting a firearm in the locked trunk of a vehicle constitute 'carrying' the firearm for purposes of the statute?"
- **Model**: gpt-4.1-2025-04-14
- **Trials**: 100 per prompt variation
- **Prompt Variations**: 50 different prompt styles
- **Output**: Results saved to `gpt4_1_prompt_test_results_temp_0.3/`

### 2. Historical Models Baby Stroller Test (Temperature 0.3)
- **File**: `historical_models_baby_stroller_test_temp_0.3.py`
- **Question**: "Is a 'baby stroller' a 'vehicle'?"
- **Models**: 16 historical models (9 OpenAI + 7 Anthropic)
- **Trials**: 100 per model
- **Output**: Results saved to `historical_models_baby_stroller_results_temp_0.3/`

## Key Differences from Original Tests

| Aspect | Original Tests | Temperature 0.3 Tests |
|--------|----------------|----------------------|
| **Temperature** | 0.0 (deterministic) | 0.3 (slightly variable) |
| **Output Folders** | Separate folders in root | `temperature_0.3_tests/` subfolder |
| **File Names** | Original names | Names include `_temp_0.3` suffix |
| **Data** | Preserved in original locations | New data in new locations |

## Expected Results

With temperature 0.3, you should expect:
- **More variability** in responses compared to temperature 0.0
- **Less deterministic** behavior
- **Potential differences** in the distribution of Yes/No responses
- **Interesting comparisons** between deterministic vs. slightly variable behavior

## Running the Tests

### GPT-4.1 Prompt Test
```bash
cd temperature_0.3_tests
python gpt4_1_prompt_test_temp_0.3.py
```

### Historical Models Test
```bash
cd temperature_0.3_tests
python historical_models_baby_stroller_test_temp_0.3.py
```

## Output Structure

```
temperature_0.3_tests/
├── gpt4_1_prompt_test_results_temp_0.3/
│   ├── detailed_results.csv
│   ├── prompt_summary.csv
│   ├── gpt4_1_prompt_variations_graph_temp_0.3.png
│   ├── gpt4_1_prompt_variations_horizontal_temp_0.3.png
│   └── test_report_temp_0.3.txt
├── historical_models_baby_stroller_results_temp_0.3/
│   ├── detailed_results.csv
│   ├── model_summary.csv
│   ├── historical_models_baby_stroller_graph_temp_0.3.png
│   └── test_report_temp_0.3.txt
├── gpt4_1_prompt_test_temp_0.3.py
├── historical_models_baby_stroller_test_temp_0.3.py
└── README.md
```

## Comparison Analysis

After running both sets of tests, you can compare:
- **Response consistency** between temperature 0.0 and 0.3
- **Prompt effectiveness** across different temperature settings
- **Model behavior** under different randomness levels
- **Statistical significance** of temperature effects

## Notes

- All original data and results remain intact
- These tests use the same API keys and models
- The same system messages and prompt structures are maintained
- Only the temperature parameter has been changed
- Results are clearly labeled with the temperature setting for easy comparison
