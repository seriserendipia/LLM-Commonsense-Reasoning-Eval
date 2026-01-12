"""
Q2 Async: LLM Accuracy Evaluation on Commonsense Reasoning Benchmarks (Async Version)

This script evaluates three LLMs (via OpenRouter) on five commonsense reasoning benchmarks
using asynchronous requests with Tenacity retry logic for optimal performance.

Benchmarks:
- CommonsenseQA
- HellaSwag
- PIQA
- SocialIQA
- TG-CSR

Output: 3x5 accuracy table (models as rows, datasets as columns)
"""

import os
import json
import re
import time
import asyncio
import warnings
import sys
from datetime import datetime, timedelta
from openai import AsyncOpenAI, RateLimitError
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

# Tenacity imports for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Import data loading functions
from load_data import get_formatted_validation_data
from api_key_loader import load_openrouter_api_key

# ============================================================================
# Suppress Windows asyncio warnings
# ============================================================================

# Suppress "Event loop is closed" warnings on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Suppress RuntimeWarning for unclosed resources
warnings.filterwarnings("ignore", category=ResourceWarning)

# ============================================================================
# Configuration
# ============================================================================

# OpenRouter API Configuration
OPENROUTER_API_KEY = load_openrouter_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to evaluate
MODELS = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-lite-001",
    "qwen/qwen-2.5-72b-instruct"
]

# Datasets to evaluate
DATASETS = [
    "CommonsenseQA",
    "HellaSwag", 
    "PIQA",
    "SocialIQA",
    "TG-CSR"
]

# Evaluation parameters
MAX_SAMPLES = 0  # Set to 0 to evaluate all samples, or N to evaluate first N samples
MAX_TOKENS = 20  # Maximum tokens to generate in LLM response

# Async & Retry Configuration
MAX_CONCURRENT_REQUESTS = 500  # Maximum number of concurrent API requests
MAX_RETRIES = 3  # Maximum number of retries for rate limit errors (429)

# Cache settings
USE_CACHE = True  # Set to False to ignore cache and re-evaluate everything
CACHE_FILE = "Q2_async_evaluation_cache.json"  # Results cache file

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Print Configuration
# ============================================================================

print("="*80)
print("LLM EVALUATION ON COMMONSENSE REASONING BENCHMARKS (ASYNC VERSION)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Models: {len(MODELS)}")
for model in MODELS:
    print(f"    - {model}")
print(f"  Datasets: {len(DATASETS)}")
for dataset in DATASETS:
    print(f"    - {dataset}")
print(f"  Max samples per dataset: {'All' if MAX_SAMPLES == 0 else MAX_SAMPLES}")
print(f"  Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
print(f"  Max retries: {MAX_RETRIES}")
print(f"  Use cache: {USE_CACHE}")
print(f"  Cache file: {CACHE_FILE}")

# ============================================================================
# Helper Functions
# ============================================================================

def load_cache():
    """Load evaluation cache from file if it exists and USE_CACHE is True."""
    if not USE_CACHE:
        print(f"\n⚠ Cache disabled (USE_CACHE=False), starting fresh evaluation")
        return {}
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"\n✓ Loaded cache from {CACHE_FILE} ({len(cache)} entries)")
            return cache
        except Exception as e:
            print(f"\n⚠ Error loading cache: {e}")
            return {}
    else:
        print(f"\n⚠ No cache file found, starting fresh evaluation")
    return {}


def save_cache(cache):
    """Save evaluation cache to file (only if USE_CACHE is True)."""
    if not USE_CACHE:
        return  # Don't save cache if caching is disabled
    
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"✓ Cache saved to {CACHE_FILE}")
    except Exception as e:
        print(f"⚠ Error saving cache: {e}")


def build_prompt(question: str, choices: List[str], is_multi_choice: bool = False) -> str:
    """
    Build a prompt for the LLM to answer a multiple-choice question.
    
    Args:
        question (str): The question text
        choices (List[str]): List of answer choices
        is_multi_choice (bool): If True, allow selecting multiple answers (for TG-CSR)
    
    Returns:
        str: Formatted prompt
    """
    # Create choice labels (A, B, C, D, E, ...)
    choice_labels = [chr(65 + i) for i in range(len(choices))]  # A, B, C, ...
    
    # Format choices
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    # Build prompt - different for single-choice vs multi-choice
    if is_multi_choice:
        # For TG-CSR: allow multiple answers or none
        prompt = f"""Question: {question}

Choices:
{choices_text}

Please select ALL correct answers from the choices above. You may select multiple answers, one answer, or none. Reply with ONLY the letter(s) separated by commas (e.g., "A", "A,B", "A,B,C", or "NONE" if no answer is correct). Do not include any explanation or additional text."""
    else:
        # For other datasets: single answer only
        prompt = f"""Question: {question}

Choices:
{choices_text}

Please select the best answer from the choices above. Reply with ONLY the letter (A, B, C, etc.) of your chosen answer. Do not include any explanation or additional text."""
    
    return prompt


# ============================================================================
# Async LLM Call with Tenacity Retry
# ============================================================================

@retry(
    retry=retry_if_exception_type(RateLimitError),  # Only retry on 429 errors
    wait=wait_random_exponential(min=1, max=60),    # Exponential backoff: 1s → 60s
    stop=stop_after_attempt(MAX_RETRIES),           # Stop after MAX_RETRIES attempts
    before_sleep=before_sleep_log(logger, logging.WARNING)  # Log before sleeping
)
async def call_llm_async(client: AsyncOpenAI, model: str, prompt: str) -> Tuple[str, bool, str]:
    """
    Call the LLM via OpenRouter API asynchronously with Tenacity retry logic.
    
    Args:
        client (AsyncOpenAI): Async OpenAI client
        model (str): Model identifier
        prompt (str): Input prompt
    
    Returns:
        tuple: (response_text, was_truncated, error_message) where:
            - response_text: str, LLM response text or None if failed
            - was_truncated: bool, True if response was truncated due to max_tokens
            - error_message: str, error details if failed, empty string if successful
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )
        
        # Check if response is valid
        if response is None or not response.choices:
            error_message = "Empty or invalid response from API"
            logger.error(f"Error calling {model}: {error_message}")
            return (None, False, error_message)
        
        # Extract response
        message_content = response.choices[0].message.content
        if message_content is None:
            error_message = "Response content is None"
            logger.error(f"Error calling {model}: {error_message}")
            return (None, False, error_message)
        
        message_content = message_content.strip()
        
        # Check if response was truncated
        finish_reason = response.choices[0].finish_reason
        was_truncated = (finish_reason == "length")
        
        return (message_content, was_truncated, "")
        
    except RateLimitError as e:
        # Re-raise RateLimitError so Tenacity can retry
        logger.warning(f"Rate limit hit for model {model}, Tenacity will retry...")
        raise
    except Exception as e:
        # Check if this is a fatal error (403 API key limit exceeded)
        error_str = str(e)
        error_type = type(e).__name__
        
        if "403" in error_str or "PermissionDenied" in error_type or "Key limit exceeded" in error_str:
            # Fatal error: API key limit exceeded
            logger.error(f"❌ FATAL ERROR: API key limit exceeded for model {model}")
            logger.error(f"Error details: {error_type}: {error_str}")
            raise  # Re-raise to stop evaluation
        else:
            # Other errors (temporary network issues, etc.), log and return None with error message
            error_message = f"{error_type}: {error_str}"
            logger.error(f"Error calling {model}: {error_message}")
            return (None, False, error_message)


def parse_answer(llm_response: str, num_choices: int, is_multi_choice: bool = False, was_truncated: bool = False):
    """
    Parse the LLM's response to extract the answer choice index/indices.
    
    Args:
        llm_response (str): Raw response from LLM
        num_choices (int): Number of available choices
        is_multi_choice (bool): If True, parse multiple answers (for TG-CSR)
        was_truncated (bool): If True, response was truncated due to max_tokens
    
    Returns:
        For single-choice (is_multi_choice=False):
            int: Answer index (0-based), or -1 if parsing failed
        For multi-choice (is_multi_choice=True):
            list of int: List of answer indices (0-based), or [-1] if parsing failed
                        Empty list [] means "NONE" (no correct answer)
    """
    if llm_response is None:
        return [-1] if is_multi_choice else -1
    
    # Clean up response
    response = llm_response.strip().upper()
    
    # Check if response was truncated and contains thinking keywords
    if was_truncated:
        thinking_keywords = [
            "WE NEED TO", "LET'S THINK", "LET ME ANALYZE",
            "FIRST, WE SHOULD", "TO SOLVE THIS", "LET'S BREAK DOWN",
            "PARSING THE QUESTION", "UNDERSTANDING THE", "I NEED TO",
            "TO ANSWER THIS", "FIRST, LET", "WE SHOULD"
        ]
        
        if any(keyword in response for keyword in thinking_keywords):
            return [-1] if is_multi_choice else -1
    
    if is_multi_choice:
        # Multi-choice parsing (for TG-CSR)
        # Check if response is "NONE" (no correct answer)
        if "NONE" in response:
            return []  # Empty list means no correct answer
        
        # Extract all letters from the response
        letters = re.findall(r'[A-Z]', response)
        
        if not letters:
            return [-1]  # Parsing failed
        
        # Convert letters to indices and remove duplicates (order doesn't matter)
        indices = []
        for letter in letters:
            idx = ord(letter) - ord('A')
            if 0 <= idx < num_choices and idx not in indices:
                indices.append(idx)
        
        if not indices:
            return [-1]  # All letters were invalid
        
        return sorted(indices)  # Return sorted list for consistent comparison
    
    else:
        # Single-choice parsing (for other datasets)
        letter = None
        
        # Try to find a single letter (A, B, C, etc.)
        # Pattern 1: Just the letter
        if len(response) == 1 and response.isalpha():
            letter = response
        else:
            # Pattern 2: Letter followed by punctuation (e.g., "A.", "A)", "A:")
            match = re.search(r'^([A-Z])[\.\)\:]', response)
            if match:
                letter = match.group(1)
            else:
                # Pattern 3: First letter in the response
                match = re.search(r'([A-Z])', response)
                if match:
                    letter = match.group(1)
        
        if letter is None:
            return -1
        
        # Convert letter to index (A=0, B=1, C=2, ...)
        answer_index = ord(letter) - ord('A')
        
        # Validate index is within valid range
        if 0 <= answer_index < num_choices:
            return answer_index
        else:
            return -1


# ============================================================================
# Async Evaluation Functions
# ============================================================================

async def evaluate_model_on_dataset_async(
    model_name: str,
    dataset_name: str,
    samples: List[Dict],
    cache: Dict,
    progress_bar=None,
    overall_question_idx=None
) -> Dict:
    """
    Evaluate a single model on a single dataset asynchronously with concurrent requests.
    
    Args:
        model_name (str): Model identifier
        dataset_name (str): Dataset name
        samples (List[Dict]): List of samples in unified format
        cache (Dict): Cache dictionary for storing results
        progress_bar: Optional tqdm progress bar for overall progress
        overall_question_idx: Optional dict to track overall question index
    
    Returns:
        Dict: Evaluation results containing accuracy, detailed stats, and timing info
    """
    cache_key = f"{model_name}::{dataset_name}"
    
    # Check if already evaluated (only if USE_CACHE is True)
    if USE_CACHE and cache_key in cache and len(cache[cache_key].get('results', [])) == len(samples):
        print(f"  ✓ Using cached results for {model_name} on {dataset_name}")
        if progress_bar and overall_question_idx is not None:
            overall_question_idx['idx'] += len(samples)
            progress_bar.update(len(samples))
        return cache[cache_key]
    
    if not USE_CACHE and cache_key in cache:
        print(f"  ⚠ Cache exists but ignored (USE_CACHE=False), re-evaluating...")
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print()
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Initialize results and timing
    results = []
    correct_count = 0
    failed_count = 0
    length_limited_count = 0
    question_times = []
    
    dataset_start_time = time.time()
    
    # Define async task for processing one question
    async def process_one_question(sample, idx):
        nonlocal correct_count, failed_count, length_limited_count
        
        question_start_time = time.time()
        
        async with semaphore:  # Limit concurrent requests
            # Determine if this is multi-choice question
            is_multi = sample.get('is_multi_choice', False)
            
            # Build prompt
            prompt = build_prompt(sample['question'], sample['choices'], is_multi_choice=is_multi)
            
            # Call LLM
            llm_response, was_truncated, error_message = await call_llm_async(client, model_name, prompt)
            
            # Parse answer
            predicted_answer = parse_answer(llm_response, len(sample['choices']), is_multi_choice=is_multi, was_truncated=was_truncated)
            
            # Check correctness (handle both single-choice and multi-choice)
            if is_multi:
                # Multi-choice: compare sets (order doesn't matter)
                # predicted_answer is list, correct_answer is list
                if predicted_answer == [-1]:
                    is_correct = False
                    parse_failed = True
                else:
                    # Convert both to sorted lists for comparison
                    pred_set = sorted(predicted_answer)
                    correct_set = sorted(sample['correct_answer'])
                    is_correct = (pred_set == correct_set)
                    parse_failed = False
            else:
                # Single-choice: simple integer comparison
                is_correct = (predicted_answer == sample['correct_answer'])
                parse_failed = (predicted_answer == -1)
            
            # Mutually exclusive classification: correct, wrong, or length_limited
            if was_truncated:
                # Length limited takes priority (regardless of correct/wrong/failed)
                length_limited_count += 1
            elif is_correct:
                correct_count += 1
            else:
                # Everything else is wrong (including parse_failed if not truncated)
                failed_count += 1
            
            # Calculate time for this question
            question_end_time = time.time()
            question_time = question_end_time - question_start_time
            question_times.append(question_time)
            
            # Update overall progress bar if provided
            if progress_bar and overall_question_idx is not None:
                overall_question_idx['idx'] += 1
                avg_time = sum(question_times) / len(question_times)
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'avg': f'{avg_time:.2f}s/q',
                    'model': model_name.split('/')[-1][:15]
                })
            
            # Store result
            result = {
                'sample_index': idx,
                'question': sample['question'][:100],
                'choices': sample['choices'],
                'correct_answer': sample['correct_answer'],
                'predicted_answer': predicted_answer,
                'llm_response': llm_response,
                'is_correct': is_correct,
                'parse_failed': parse_failed,
                'length_limited': was_truncated,
                'time_seconds': question_time,
                'is_multi_choice': is_multi,
                'error_message': error_message if error_message else None  # Add error info
            }
            
            return result
    
    # Create all tasks
    tasks = [process_one_question(sample, idx) for idx, sample in enumerate(samples)]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Calculate timing statistics
    dataset_end_time = time.time()
    dataset_total_time = dataset_end_time - dataset_start_time
    avg_time_per_question = sum(question_times) / len(question_times) if question_times else 0.0
    
    # Calculate accuracy and statistics
    total = len(samples)
    accuracy = (correct_count / total * 100) if total > 0 else 0.0
    length_limited_rate = (length_limited_count / total * 100) if total > 0 else 0.0
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {total - correct_count - failed_count}")
    print(f"  Parse failed: {failed_count}")
    print(f"  Length limited: {length_limited_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Length limited rate: {length_limited_rate:.2f}%")
    print(f"\n  ⏱️  Timing:")
    print(f"     Total time: {dataset_total_time:.2f}s ({dataset_total_time/60:.2f} min)")
    print(f"     Avg per question: {avg_time_per_question:.2f}s")
    print(f"     Min/Max: {min(question_times):.2f}s / {max(question_times):.2f}s")
    print(f"     Speedup: {len(samples) * avg_time_per_question / dataset_total_time:.1f}x vs serial")
    
    # Store in cache
    evaluation_result = {
        'model': model_name,
        'dataset': dataset_name,
        'total_samples': total,
        'correct_count': correct_count,
        'failed_count': failed_count,
        'length_limited_count': length_limited_count,
        'accuracy': accuracy,
        'length_limited_rate': length_limited_rate,
        'total_time_seconds': dataset_total_time,
        'avg_time_per_question': avg_time_per_question,
        'question_times': question_times,
        'results': results
    }
    
    cache[cache_key] = evaluation_result
    save_cache(cache)
    
    return evaluation_result


# ============================================================================
# Results Printing Functions (Reused from Q2.py)
# ============================================================================

def print_results_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print the final 3x5 results table.
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """
    print(f"\n{'='*80}")
    print("FINAL RESULTS: ACCURACY TABLE (%) ")
    print(f"{'='*80}")
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    print(header)
    print("-" * 80)
    
    # Print each model's results
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                accuracy = all_results[model][dataset]['accuracy']
                row += f"{accuracy:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print("-" * 80)


def print_detailed_statistics_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print a detailed statistics table with three metrics:
    1. Accuracy (%)
    2. Length-Limited Rate (%) - responses truncated due to max_tokens
    3. Wrong Answer Rate (%) - incorrect answers not due to length limit
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """

    print(f"\n{'='*120}")
    print(f"DETAILED STATISTICS: ACCURACY, LENGTH-LIMITED RATE & WRONG ANSWER RATE (max_tokens={MAX_TOKENS})")
    print(f"{'='*120}")
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    print(header)
    print("=" * 120)
    
    # Metric 1: Accuracy
    print(f"{'Accuracy (%):':<40}")
    print("-" * 120)
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                accuracy = all_results[model][dataset]['accuracy']
                row += f"{accuracy:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 120)
    
    # Metric 2: Length-Limited Rate
    print(f"{'Length-Limited Rate (%):':<40}")
    print(f"{'(Responses truncated by max_tokens)':<40}")
    print("-" * 120)
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                rate = all_results[model][dataset]['length_limited_rate']
                row += f"{rate:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 120)
    
    # Metric 3: Wrong Answer Rate (mutually exclusive with correct and length_limited)
    print(f"{'Wrong Answer Rate (%):':<40}")
    print(f"{'(Incorrect, mutually exclusive)':<40}")
    print("-" * 120)
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                total = all_results[model][dataset]['total_samples']
                failed = all_results[model][dataset]['failed_count']
                
                # Wrong answers (mutually exclusive: correct + wrong + length_limited = 100%)
                wrong_rate = (failed / total * 100) if total > 0 else 0.0
                row += f"{wrong_rate:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 120)


def print_timing_statistics_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print a timing statistics table showing time spent on each model-dataset combination.
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """
    print(f"\n{'='*120}")
    print(f"TIMING STATISTICS")
    print(f"{'='*120}")
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    header += f"{'Total Time':>18}"
    print(header)
    print("=" * 120)
    
    # Dataset time per model (in seconds)
    print(f"{'Dataset Time (seconds):':<40}")
    print("-" * 120)
    model_totals = {}
    for model in MODELS:
        row = f"{model:<40}"
        model_total = 0.0
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                dataset_time = all_results[model][dataset]['total_time_seconds']
                row += f"{dataset_time:>14.2f}s"
                model_total += dataset_time
            else:
                row += f"{'N/A':>15}"
        model_totals[model] = model_total
        row += f"{model_total:>17.2f}s"
        print(row)
    
    print("=" * 120)
    
    # Average time per question for each model-dataset
    print(f"{'Avg Time per Question (seconds):':<40}")
    print("-" * 120)
    for model in MODELS:
        row = f"{model:<40}"
        avg_times = []
        for dataset in DATASETS:
            if model in all_results and dataset in all_results[model]:
                avg_time = all_results[model][dataset]['avg_time_per_question']
                row += f"{avg_time:>14.2f}s"
                avg_times.append(avg_time)
            else:
                row += f"{'N/A':>15}"
        # Overall average across datasets
        if avg_times:
            overall_avg = sum(avg_times) / len(avg_times)
            row += f"{overall_avg:>17.2f}s"
        print(row)
    
    print("=" * 120)
    
    # Time per dataset (sum across all models)
    print(f"{'Total Time per Dataset (all models):':<40}")
    print("-" * 120)
    row = f"{'Sum across models':<40}"
    dataset_totals = {}
    for dataset in DATASETS:
        dataset_total = sum(
            all_results[model][dataset]['total_time_seconds']
            for model in MODELS
            if model in all_results and dataset in all_results[model]
        )
        dataset_totals[dataset] = dataset_total
        row += f"{dataset_total:>14.2f}s"
    
    grand_total = sum(model_totals.values())
    row += f"{grand_total:>17.2f}s"
    print(row)
    
    # Convert to minutes
    row = f"{'(in minutes)':<40}"
    for dataset in DATASETS:
        row += f"{dataset_totals.get(dataset, 0)/60:>14.2f}m"
    row += f"{grand_total/60:>17.2f}m"
    print(row)
    
    print("=" * 120)


# ============================================================================
# Main Evaluation Loop (Async)
# ============================================================================

async def main_async():
    """Main async evaluation function with overall timing."""
    overall_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("STARTING ASYNC EVALUATION")
    print(f"{'='*80}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Total combinations: {len(MODELS) * len(DATASETS)}")
    
    # Load cache
    cache = load_cache()
    
    # Store all results: model -> dataset -> results
    all_results = {}
    
    # Calculate total work - count total questions across all datasets
    print("\nCalculating total questions...")
    # Load all datasets
    all_datasets = {}
    for dataset_name in DATASETS:
        all_datasets[dataset_name] = get_formatted_validation_data(dataset_name, max_samples=MAX_SAMPLES)
    
    total_questions = 0
    dataset_sample_counts = {}
    for dataset_name in DATASETS:
        count = len(all_datasets[dataset_name])
        dataset_sample_counts[dataset_name] = count
        total_questions += count
        print(f"  {dataset_name}: {count} samples")
    
    total_questions_all_models = total_questions * len(MODELS)
    print(f"Total questions to process: {total_questions_all_models} ({total_questions} questions × {len(MODELS)} models)")
    
    # Create overall progress bar
    print(f"\n{'='*80}")
    print("EVALUATION PROGRESS")
    print(f"{'='*80}\n")
    
    overall_pbar = tqdm(
        total=total_questions_all_models,
        desc="Overall Progress",
        unit="q",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    )
    
    overall_question_idx = {'idx': 0}
    
    # Evaluate each model on each dataset
    try:
        for model_idx, model in enumerate(MODELS):
            print(f"\n{'='*80}")
            print(f"Model {model_idx + 1}/{len(MODELS)}: {model}")
            print(f"{'='*80}")
            
            if model not in all_results:
                all_results[model] = {}
            
            for dataset_idx, dataset_name in enumerate(DATASETS):
                print(f"\nDataset {dataset_idx + 1}/{len(DATASETS)}: {dataset_name}")
                
                # Get dataset samples (already limited by MAX_SAMPLES during loading)
                dataset = all_datasets[dataset_name]
                
                try:
                    # Evaluate model on dataset (async)
                    result = await evaluate_model_on_dataset_async(
                        model,
                        dataset_name,
                        dataset,
                        cache,
                        overall_pbar,
                        overall_question_idx
                    )
                    
                    all_results[model][dataset_name] = result
                    
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    # Check if this is a fatal error (403 API key limit exceeded)
                    if "403" in error_str or "PermissionDenied" in error_type or "Key limit exceeded" in error_str:
                        # Fatal error: Save progress and exit
                        print(f"\n\n{'='*80}")
                        print(f"❌ FATAL ERROR: API KEY LIMIT EXCEEDED")
                        print(f"{'='*80}")
                        print(f"\n📍 Error occurred at:")
                        print(f"   Model: {model} ({model_idx + 1}/{len(MODELS)})")
                        print(f"   Dataset: {dataset_name} ({dataset_idx + 1}/{len(DATASETS)})")
                        print(f"   Error type: {error_type}")
                        print(f"   Error message: {error_str}")
                        
                        print(f"\n📊 Progress Summary:")
                        print(f"   Total combinations: {len(MODELS)} models × {len(DATASETS)} datasets = {len(MODELS) * len(DATASETS)}")
                        
                        # Count completed combinations
                        completed_count = 0
                        for m in MODELS:
                            if m in all_results:
                                completed_count += len(all_results[m])
                        
                        print(f"   Completed: {completed_count}/{len(MODELS) * len(DATASETS)} combinations")
                        print(f"   Remaining: {len(MODELS) * len(DATASETS) - completed_count} combinations")
                        
                        # Show completed combinations
                        print(f"\n✅ Completed combinations:")
                        for m_idx, m in enumerate(MODELS):
                            if m in all_results and all_results[m]:
                                completed_datasets = list(all_results[m].keys())
                                print(f"   {m}:")
                                for d in completed_datasets:
                                    acc = all_results[m][d]['accuracy']
                                    samples = all_results[m][d]['total_samples']
                                    print(f"     ✓ {d}: {acc:.2f}% ({samples} samples)")
                        
                        # Show what's remaining
                        print(f"\n⏸️  Incomplete combinations:")
                        for m_idx, m in enumerate(MODELS):
                            remaining_datasets = [d for d in DATASETS if m not in all_results or d not in all_results.get(m, {})]
                            if remaining_datasets:
                                print(f"   {m}: {', '.join(remaining_datasets)}")
                        
                        print(f"\n💾 Saving current progress to cache...")
                        save_cache(cache)
                        print(f"   ✓ Progress saved to {CACHE_FILE}")
                        
                        print(f"\n💡 How to resume:")
                        print(f"   1. Check your API key at: https://openrouter.ai/settings/keys")
                        print(f"   2. Add credits or update to a new API key")
                        print(f"   3. Update OPENROUTER_API_KEY in this script (line ~55)")
                        print(f"   4. Run the script again with same settings:")
                        print(f"      - USE_CACHE = True (current: {USE_CACHE})")
                        print(f"      - MAX_SAMPLES = {MAX_SAMPLES}")
                        print(f"   5. The script will automatically skip completed combinations")
                        print(f"      and continue from where it stopped")
                        
                        print(f"\n{'='*80}\n")
                        
                        overall_pbar.close()
                        raise SystemExit(1)  # Exit with error code
                    else:
                        # Other errors: log and continue
                        logger.error(f"Error evaluating {model} on {dataset_name}: {e}")
                        print(f"⚠️  Error occurred but continuing: {e}")
                        continue
                        
    except KeyboardInterrupt:
        # User interrupted: save progress
        print(f"\n\n{'='*80}")
        print(f"⚠️  INTERRUPTED BY USER")
        print(f"{'='*80}")
        print(f"\n💾 Saving current progress...")
        save_cache(cache)
        print(f"   ✓ Progress saved to {CACHE_FILE}")
        
        # Count completed
        completed_count = sum(len(all_results.get(m, {})) for m in MODELS)
        print(f"\n📊 Progress at interruption:")
        print(f"   Completed: {completed_count}/{len(MODELS) * len(DATASETS)} combinations")
        print(f"\n💡 Run the script again to resume from where you stopped.")
        print(f"{'='*80}\n")
        
        overall_pbar.close()
        raise
    
    # Close progress bar
    overall_pbar.close()
    
    # Calculate total time
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    end_datetime = datetime.now()
    
    # Print final results tables
    print_results_table(all_results)
    print_detailed_statistics_table(all_results)
    print_timing_statistics_table(all_results)
    
    # Print overall timing summary
    print(f"\n{'='*80}")
    print("OVERALL TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Start time:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_time:.2f}s ({total_time/60:.2f} min / {total_time/3600:.2f} hours)")
    print(f"Total questions processed: {overall_question_idx['idx']}")
    if overall_question_idx['idx'] > 0:
        print(f"Avg per question: {total_time/overall_question_idx['idx']:.2f}s")
    print(f"{'='*80}")
    
    # Save final results to a separate file
    results_file = "Q2_async_final_results.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Final results saved to {results_file}")
    except Exception as e:
        print(f"\n⚠ Error saving final results: {e}")
    
    print(f"\n{'='*80}")
    print("ASYNC EVALUATION COMPLETED!")
    print(f"{'='*80}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Synchronous wrapper to run async main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
