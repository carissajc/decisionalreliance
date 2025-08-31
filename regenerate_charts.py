#!/usr/bin/env python3
"""
Regenerate Charts for Trial Run
This script regenerates all charts using the updated chart generator
"""

import asyncio
import logging
from pathlib import Path
import pandas as pd

from chart_generator import ChartGenerator
from experiment_config import *

def regenerate_all_charts():
    """Regenerate all charts for the trial run"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    chart_gen = ChartGenerator()
    
    logger.info("Regenerating all charts for trial run...")
    
    for i in range(TRIAL_QUESTIONS):
        question_dir = FULL_EXPERIMENT_DIR / f"question_{i+1}"
        results_file = question_dir / f"results_question_{i+1}.csv"
        
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                results = df.to_dict('records')
                
                if not results:
                    logger.warning(f"No results found for question {i+1}")
                    continue
                
                # Get question data
                question_data = {
                    'question_index': i + 1,
                    'question': results[0]['question_label']
                }
                
                # Generate updated charts
                chart_gen.generate_question_charts(question_data, results, i)
                logger.info(f"Regenerated charts for question {i+1}")
                
            except Exception as e:
                logger.error(f"Error regenerating charts for question {i+1}: {e}")
        else:
            logger.warning(f"Results file not found for question {i+1}: {results_file}")
    
    logger.info("Chart regeneration completed!")

if __name__ == "__main__":
    regenerate_all_charts()

