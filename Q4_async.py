"""
Q4 Async: LLM Evaluation with "I don't know" Option (Async Version)

This script evaluates three LLMs on five commonsense reasoning benchmarks
with an additional "I don't know" option added to each question.
Uses asynchronous requests with Tenacity retry logic for optimal performance.

Experiment Design:
- Add "I don't know" as the last option for all questions
- CommonsenseQA: 5 options → 6 options (5 original + "I don't know")
- HellaSwag: 4 options → 5 options
- PIQA: 2 options → 3 options
- SocialIQA: 3 options → 4 options
- TG-CSR: 2 options → 3 options

Analysis:
(a) For questions LLM got wrong in Q2, how many now say "I don't know"?
(b) For questions LLM got right in Q2, how many now say "I don't know"?

Output: Absolute numbers (not percentages)
"""

import os
import json
import re
import time
import asyncio
import warnings
import sys
from datetime import datetime
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

# Models to evaluate (same as Q2)
MODELS = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-lite-001",
    "qwen/qwen-2.5-72b-instruct"
]

# Datasets to evaluate (all 5 datasets, same as Q2)
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
MAX_CONCURRENT_REQUESTS = 50  # Maximum number of concurrent API requests
MAX_RETRIES = 3  # Maximum number of retries for rate limit errors (429)

# Cache settings
USE_CACHE = False  # Set to False to ignore cache and re-evaluate everything
CACHE_FILE = "Q4_async_evaluation_cache.json"  # Q4 specific cache file
RESULTS_FILE = "Q4_async_final_results.json"  # Q4 specific results file
Q2_RESULTS_FILE = "Q2_async_final_results.json"  # Q2 async results for comparison

# "I don't know" option text
IDK_OPTION = "I don't know"

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
print("Q4: LLM EVALUATION WITH 'I DON'T KNOW' OPTION (ASYNC VERSION)")
print("="*80)
print(f"\nExperiment Design:")
print(f"  Add '{IDK_OPTION}' as the last option for all questions")
print(f"  - CommonsenseQA: 5 options → 6 options")
print(f"  - HellaSwag: 4 options → 5 options")
print(f"  - PIQA: 2 options → 3 options")
print(f"  - SocialIQA: 3 options → 4 options")
print(f"  - TG-CSR: 2 options → 3 options")
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
print(f"  Q2 results file: {Q2_RESULTS_FILE}")

# ============================================================================
# Q4 Specific: Add "I don't know" Option
# ============================================================================

def add_idk_option(sample):
    """
    Add "I don't know" as the last option to a sample.
    
    Args:
        sample (dict): Original sample
            - question: str
            - choices: list of str (N options)
            - correct_answer: int (0-based index)
    
    Returns:
        dict: Modified sample with "I don't know" added
            - question: str (unchanged)
            - choices: list of str (N+1 options, last one is "I don't know")
            - correct_answer: int (unchanged, since "I don't know" is added at the end)
            - idk_index: int (index of "I don't know", which is N)
    """
    new_choices = sample['choices'] + [IDK_OPTION]
    
    return {
        'question': sample['question'],
        'choices': new_choices,
        'correct_answer': sample['correct_answer'],  # Index unchanged
        'idk_index': len(new_choices) - 1  # "I don't know" is the last option
    }


def get_formatted_validation_data_with_idk(dataset_name, max_samples=0, print_examples=False):
    """
    Get validation data with "I don't know" option added.
    
    This function:
    1. Loads validation data using the same method as Q2 (to ensure same order)
    2. Adds "I don't know" as the last option to each sample
    3. Returns list of modified samples
    
    Args:
        dataset_name (str): Name of the dataset
        max_samples (int): Maximum number of samples (0 = all)
        print_examples (bool): Whether to print example transformations
    
    Returns:
        list of dict: List of samples with "I don't know" option
    """
    # Step 1: Load original validation data (same as Q2)
    original_samples = get_formatted_validation_data(dataset_name, max_samples=max_samples)
    
    if not original_samples:
        print(f"❌ No samples loaded for {dataset_name}")
        return []
    
    # Step 2: Add "I don't know" option to each sample
    samples_with_idk = []
    for sample in original_samples:
        modified_sample = add_idk_option(sample)
        samples_with_idk.append(modified_sample)
    
    original_num_choices = len(original_samples[0]['choices'])
    new_num_choices = len(samples_with_idk[0]['choices'])
    
    if print_examples:
        print(f"✓ Added '{IDK_OPTION}' option to {len(samples_with_idk)} samples")
        print(f"✓ Choices per question: {original_num_choices} → {new_num_choices}")
    
    return samples_with_idk


# ============================================================================
# Load Q2 Results for Comparison
# ============================================================================

def load_q2_results():
    """
    Load Q2 async evaluation results for comparison.
    
    Returns:
        dict: Q2 results {model -> dataset -> results}, or None if file not found
    """
    if not os.path.exists(Q2_RESULTS_FILE):
        print(f"\n⚠ WARNING: Q2 async results file not found: {Q2_RESULTS_FILE}")
        print(f"  Comparison with Q2 will not be available.")
        print(f"  Please run Q2_async.py first to generate Q2 async results.")
        return None
    
    try:
        with open(Q2_RESULTS_FILE, 'r', encoding='utf-8') as f:
            q2_results = json.load(f)
        print(f"\n✓ Loaded Q2 async results from {Q2_RESULTS_FILE}")
        
        # Print Q2 statistics
        print(f"\n  Q2 Async Results Summary:")
        for model in MODELS:
            if model in q2_results:
                print(f"    {model}:")
                for dataset in DATASETS:
                    if dataset in q2_results[model]:
                        total = q2_results[model][dataset]['total_samples']
                        correct = q2_results[model][dataset]['correct_count']
                        accuracy = q2_results[model][dataset]['accuracy']
                        print(f"      - {dataset}: {correct}/{total} correct ({accuracy:.2f}%)")
        
        return q2_results
    except Exception as e:
        print(f"\n⚠ Error loading Q2 results: {e}")
        return None


def compare_with_q2(q4_results, q2_results, model, dataset):
    """
    Compare Q4 results with Q2 results for a specific model-dataset combination.
    
    Args:
        q4_results (dict): Q4 evaluation results
        q2_results (dict): Q2 evaluation results
        model (str): Model name
        dataset (str): Dataset name
    
    Returns:
        dict: Comparison statistics
            - q2_wrong_to_idk: Q2 wrong → Q4 "I don't know"
            - q2_correct_to_idk: Q2 correct → Q4 "I don't know"
            - q2_wrong_to_correct: Q2 wrong → Q4 correct
            - q2_correct_to_wrong: Q2 correct → Q4 wrong (not IDK)
            - both_correct: Both Q2 and Q4 correct
            - both_wrong: Both Q2 and Q4 wrong (Q4 not IDK)
    """
    if q2_results is None:
        return None
    
    if model not in q2_results or dataset not in q2_results[model]:
        print(f"⚠ Q2 results not found for {model} on {dataset}")
        return None
    
    if model not in q4_results or dataset not in q4_results[model]:
        print(f"⚠ Q4 results not found for {model} on {dataset}")
        return None
    
    q4_result = q4_results[model][dataset]
    q2_result = q2_results[model][dataset]
    
    # Verify same number of samples
    if len(q4_result['results']) != len(q2_result['results']):
        print(f"⚠ Sample count mismatch: Q4={len(q4_result['results'])}, Q2={len(q2_result['results'])}")
        return None
    
    stats = {
        'q2_wrong_to_idk': 0,
        'q2_correct_to_idk': 0,
        'q2_wrong_to_correct': 0,
        'q2_correct_to_wrong': 0,
        'both_correct': 0,
        'both_wrong': 0
    }
    
    # Compare each sample
    for idx in range(len(q4_result['results'])):
        q2_sample = q2_result['results'][idx]
        q4_sample = q4_result['results'][idx]
        
        q2_correct = q2_sample['is_correct']
        q4_correct = q4_sample['is_correct']
        q4_is_idk = q4_sample.get('is_idk', False)
        
        if q2_correct:
            # Q2 was correct
            if q4_is_idk:
                stats['q2_correct_to_idk'] += 1
            elif q4_correct:
                stats['both_correct'] += 1
            else:
                stats['q2_correct_to_wrong'] += 1
        else:
            # Q2 was wrong
            if q4_is_idk:
                stats['q2_wrong_to_idk'] += 1
            elif q4_correct:
                stats['q2_wrong_to_correct'] += 1
            else:
                stats['both_wrong'] += 1
    
    return stats


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
            print(f"\n✓ Cache loaded from {CACHE_FILE} ({len(cache)} entries)")
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
        return
    
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"✓ Cache saved to {CACHE_FILE}")
    except Exception as e:
        print(f"⚠ Error saving cache: {e}")


def build_prompt(question: str, choices: List[str], is_multi_choice: bool = False) -> str:
    """
    Build a prompt for the LLM to answer a multiple-choice question.
    Works with any number of choices (including "I don't know").
    
    Args:
        question (str): The question text
        choices (List[str]): List of answer choices (including "I don't know" as last option)
        is_multi_choice (bool): If True, allow selecting multiple answers (for TG-CSR)
    
    Returns:
        str: Formatted prompt
    """
    # Create choice labels (A, B, C, D, E, F, ... depending on number of choices)
    choice_labels = [chr(65 + i) for i in range(len(choices))]
    
    # Format choices
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    # Build prompt - different for single-choice vs multi-choice
    if is_multi_choice:
        # For TG-CSR: allow multiple answers or "I don't know"
        prompt = f"""Question: {question}

Choices:
{choices_text}

Please select ALL correct answers from the choices above. You may select multiple answers, one answer, or choose "I don't know" if you're uncertain. Reply with ONLY the letter(s) separated by commas (e.g., "A", "A,B", "A,B,C"). Do not include any explanation or additional text."""
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
        tuple: (response_text, was_truncated, error_message)
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
    Supports up to 26 choices (A-Z).
    
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
        
        # Try to find a single letter
        if len(response) == 1 and response.isalpha():
            letter = response
        else:
            # Pattern: Letter followed by punctuation
            match = re.search(r'^([A-Z])[\.\)\:]', response)
            if match:
                letter = match.group(1)
            else:
                # Pattern: Just find the first letter
                match = re.search(r'\b([A-Z])\b', response)
                if match:
                    letter = match.group(1)
        
        if letter is None:
            return -1
        
        # Convert letter to index (A=0, B=1, ...)
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
    Evaluate a single model on a single dataset asynchronously with "I don't know" option.
    
    Returns:
        Dict: Evaluation results with additional "I don't know" statistics
    """
    cache_key = f"{model_name}::{dataset_name}"
    
    # Check cache
    if USE_CACHE and cache_key in cache and len(cache[cache_key].get('results', [])) == len(samples):
        print(f"  ✓ Using cached results for {model_name} on {dataset_name}")
        if progress_bar and overall_question_idx is not None:
            overall_question_idx['idx'] += len(samples)
            progress_bar.update(len(samples))
        return cache[cache_key]
    
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
    idk_count = 0  # Count of "I don't know" responses
    length_limited_count = 0
    question_times = []
    
    dataset_start_time = time.time()
    
    # Define async task for processing one question
    async def process_one_question(sample, idx):
        nonlocal correct_count, failed_count, idk_count, length_limited_count
        
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
            
            # Check if "I don't know" was selected
            if is_multi:
                # For multi-choice, check if IDK is in the predicted set
                is_idk = (sample['idk_index'] in predicted_answer) if predicted_answer != [-1] else False
            else:
                # For single-choice, simple comparison
                is_idk = (predicted_answer == sample['idk_index'])
            
            # Check correctness
            if is_multi:
                # Multi-choice correctness check
                # Special rule: if question has no correct answer, only "I don't know" is correct
                correct_answer_list = sample['correct_answer']
                
                if len(correct_answer_list) == 0:
                    # No correct answer exists -> only IDK is correct
                    if predicted_answer == [-1]:
                        is_correct = False
                        parse_failed = True
                    else:
                        is_correct = is_idk and (predicted_answer == [sample['idk_index']])
                        parse_failed = False
                else:
                    # Has correct answers -> compare sets (IDK should not be selected)
                    if predicted_answer == [-1]:
                        is_correct = False
                        parse_failed = True
                    elif is_idk:
                        # If LLM selected IDK when there ARE correct answers, it's wrong
                        is_correct = False
                        parse_failed = False
                    else:
                        # Compare predicted set with correct answer set
                        pred_set = sorted(predicted_answer)
                        correct_set = sorted(correct_answer_list)
                        is_correct = (pred_set == correct_set)
                        parse_failed = False
            else:
                # Single-choice correctness check
                is_correct = (predicted_answer == sample['correct_answer'])
                parse_failed = (predicted_answer == -1)
            
            # Mutually exclusive classification: correct, IDK, wrong, or length_limited
            if was_truncated:
                # Length limited takes priority
                length_limited_count += 1
            elif is_correct:
                correct_count += 1
            elif is_idk:
                # Selected "I don't know" (not correct, not truncated)
                idk_count += 1
            else:
                # Everything else is wrong (including parse_failed if not truncated or IDK)
                failed_count += 1
            
            # Calculate time
            question_end_time = time.time()
            question_time = question_end_time - question_start_time
            question_times.append(question_time)
            
            # Update progress bar
            if progress_bar and overall_question_idx is not None:
                overall_question_idx['idx'] += 1
                avg_time = sum(question_times) / len(question_times)
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'avg': f'{avg_time:.2f}s/q',
                    'model': model_name.split('/')[-1][:15],
                    'idk': idk_count
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
                'is_idk': is_idk,  # New: Whether "I don't know" was selected
                'idk_index': sample['idk_index'],  # New: Index of "I don't know"
                'parse_failed': parse_failed,
                'length_limited': was_truncated,
                'time_seconds': question_time,
                'is_multi_choice': is_multi,
                'error_message': error_message if error_message else None  # Add error info
            }
            
            return result
    
    # Create tasks for all questions
    tasks = [process_one_question(sample, idx) for idx, sample in enumerate(samples)]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    dataset_end_time = time.time()
    dataset_total_time = dataset_end_time - dataset_start_time
    avg_time_per_question = sum(question_times) / len(question_times) if question_times else 0.0
    
    total = len(samples)
    accuracy = (correct_count / total * 100) if total > 0 else 0.0
    idk_rate = (idk_count / total * 100) if total > 0 else 0.0
    length_limited_rate = (length_limited_count / total * 100) if total > 0 else 0.0
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  'I don't know': {idk_count}")
    print(f"  Incorrect (not IDK): {total - correct_count - failed_count - idk_count}")
    print(f"  Parse failed: {failed_count}")
    print(f"  Length limited: {length_limited_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  'I don't know' rate: {idk_rate:.2f}%")
    print(f"\n  ⏱️  Timing:")
    print(f"     Total time: {dataset_total_time:.2f}s ({dataset_total_time/60:.2f} min)")
    print(f"     Avg per question: {avg_time_per_question:.2f}s")
    
    # Store in cache
    evaluation_result = {
        'model': model_name,
        'dataset': dataset_name,
        'total_samples': total,
        'correct_count': correct_count,
        'idk_count': idk_count,  # New
        'failed_count': failed_count,
        'length_limited_count': length_limited_count,
        'accuracy': accuracy,
        'idk_rate': idk_rate,  # New
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
# Output Tables
# ============================================================================

def print_results_table(all_results: Dict[str, Dict[str, Dict]]):
    """Print Q4 accuracy results table."""
    print(f"\n{'='*120}")
    print("Q4 ASYNC FINAL RESULTS: ACCURACY TABLE (%) - With 'I don't know' Option")
    print(f"{'='*120}")
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    print(header)
    print("-" * 120)
    
    # Print each model's results
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if dataset in all_results.get(model, {}):
                accuracy = all_results[model][dataset]['accuracy']
                row += f"{accuracy:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("-" * 120)


def print_idk_rate_table(all_results: Dict[str, Dict[str, Dict]]):
    """Print 'I don't know' selection rate table."""
    print(f"\n{'='*120}")
    print("Q4: 'I DON'T KNOW' SELECTION RATE (%) - By Model and Dataset")
    print(f"{'='*120}")
    print(f"Shows the percentage of times each model selected 'I don't know' option")
    print("-" * 120)
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    print(header)
    print("-" * 120)
    
    # Print each model's IDK rates
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if dataset in all_results.get(model, {}):
                idk_rate = all_results[model][dataset]['idk_rate']
                row += f"{idk_rate:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("-" * 120)
    
    # Print absolute counts table
    print(f"\n{'='*120}")
    print("Q4: 'I DON'T KNOW' ABSOLUTE COUNTS - By Model and Dataset")
    print(f"{'='*120}")
    print("-" * 120)
    print(header)
    print("-" * 120)
    
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if dataset in all_results.get(model, {}):
                idk_count = all_results[model][dataset]['idk_count']
                total_count = all_results[model][dataset]['total_samples']
                row += f"{idk_count}/{total_count:>12}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("-" * 120)


def print_comparison_tables(all_comparisons: Dict[str, Dict[str, Dict]]):
    """
    Print comparison tables between Q4 and Q2 async results.
    
    Answers the two key questions:
    (a) Q2 wrong → Q4 "I don't know"
    (b) Q2 correct → Q4 "I don't know"
    """
    if not all_comparisons or all(not all_comparisons.get(model) for model in MODELS):
        print(f"\n⚠ No Q2 comparison available (Q2 async results not found)")
        return
    
    print(f"\n{'='*120}")
    print("Q4 vs Q2 COMPARISON ANALYSIS (ASYNC)")
    print(f"{'='*120}")
    
    # Table (a): Q2 Wrong → Q4 "I don't know"
    print(f"\n(a) Questions LLM got WRONG in Q2, now says 'I don't know' in Q4:")
    print(f"    (Absolute numbers)")
    print("-" * 120)
    header = f"{'Model':<40}"
    for dataset in DATASETS:
        header += f"{dataset:>15}"
    print(header)
    print("-" * 120)
    
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_comparisons and dataset in all_comparisons[model]:
                count = all_comparisons[model][dataset]['q2_wrong_to_idk']
                row += f"{count:>15}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("-" * 120)
    
    # Table (b): Q2 Correct → Q4 "I don't know"
    print(f"\n(b) Questions LLM got RIGHT in Q2, now says 'I don't know' in Q4:")
    print(f"    (Absolute numbers)")
    print("-" * 120)
    print(header)
    print("-" * 120)
    
    for model in MODELS:
        row = f"{model:<40}"
        for dataset in DATASETS:
            if model in all_comparisons and dataset in all_comparisons[model]:
                count = all_comparisons[model][dataset]['q2_correct_to_idk']
                row += f"{count:>15}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("-" * 120)
    
    # Detailed transition statistics
    print(f"\n{'='*120}")
    print("DETAILED TRANSITION STATISTICS (Q2 → Q4, ASYNC)")
    print(f"{'='*120}")
    
    for model in MODELS:
        if model not in all_comparisons:
            continue
        
        print(f"\nModel: {model}")
        print("-" * 120)
        
        for dataset in DATASETS:
            if dataset not in all_comparisons[model]:
                continue
            
            stats = all_comparisons[model][dataset]
            
            print(f"\n  Dataset: {dataset}")
            print(f"    Q2 Wrong:")
            print(f"      → Q4 'I don't know': {stats['q2_wrong_to_idk']}")
            print(f"      → Q4 Correct:        {stats['q2_wrong_to_correct']}")
            print(f"      → Q4 Still Wrong:    {stats['both_wrong']}")
            print(f"    Q2 Correct:")
            print(f"      → Q4 'I don't know': {stats['q2_correct_to_idk']}")
            print(f"      → Q4 Still Correct:  {stats['both_correct']}")
            print(f"      → Q4 Wrong:          {stats['q2_correct_to_wrong']}")


# ============================================================================
# Main Evaluation Loop
# ============================================================================

async def main_async():
    """Main async evaluation function."""
    overall_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("STARTING Q4 ASYNC EVALUATION")
    print(f"{'='*80}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load Q2 async results for comparison
    q2_results = load_q2_results()
    
    # Load cache
    cache = load_cache()
    
    # Store all results
    all_results = {}
    all_comparisons = {}
    
    # Calculate total questions
    print("\nCalculating total questions...")
    total_questions = 0
    dataset_sample_counts = {}
    for dataset_name in DATASETS:
        try:
            samples = get_formatted_validation_data_with_idk(dataset_name, max_samples=MAX_SAMPLES, print_examples=False)
            dataset_sample_counts[dataset_name] = len(samples)
            total_questions += len(samples)
        except:
            dataset_sample_counts[dataset_name] = 0
    
    total_questions_all_models = total_questions * len(MODELS)
    print(f"Total questions to process: {total_questions_all_models} ({total_questions} questions × {len(MODELS)} models)")
    
    # Create progress bar
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
            all_results[model] = {}
            all_comparisons[model] = {}
            
            for dataset_idx, dataset_name in enumerate(DATASETS):
                try:
                    # Load data with "I don't know" option
                    samples = get_formatted_validation_data_with_idk(
                        dataset_name, 
                        max_samples=MAX_SAMPLES, 
                        print_examples=(model == MODELS[0] and dataset_name == DATASETS[0])
                    )
                    
                    if not samples:
                        print(f"⚠ No samples loaded for {dataset_name}, skipping...")
                        continue
                    
                    # Evaluate asynchronously
                    result = await evaluate_model_on_dataset_async(
                        model, dataset_name, samples, cache,
                        progress_bar=overall_pbar,
                        overall_question_idx=overall_question_idx
                    )
                    all_results[model][dataset_name] = result
                    
                    # Compare with Q2
                    if q2_results:
                        comparison = compare_with_q2(all_results, q2_results, model, dataset_name)
                        if comparison:
                            all_comparisons[model][dataset_name] = comparison
                    
                    print()
                    
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
                        for m in MODELS:
                            if m in all_results and all_results[m]:
                                completed_datasets = list(all_results[m].keys())
                                print(f"   {m}:")
                                for d in completed_datasets:
                                    acc = all_results[m][d]['accuracy']
                                    idk_rate = all_results[m][d].get('idk_rate', 0)
                                    samples = all_results[m][d]['total_samples']
                                    print(f"     ✓ {d}: Acc={acc:.2f}%, IDK={idk_rate:.2f}% ({samples} samples)")
                        
                        # Show what's remaining
                        print(f"\n⏸️  Incomplete combinations:")
                        for m in MODELS:
                            remaining_datasets = [d for d in DATASETS if m not in all_results or d not in all_results.get(m, {})]
                            if remaining_datasets:
                                print(f"   {m}: {', '.join(remaining_datasets)}")
                        
                        print(f"\n💾 Saving current progress to cache...")
                        save_cache(cache)
                        print(f"   ✓ Progress saved to {CACHE_FILE}")
                        
                        print(f"\n💡 How to resume:")
                        print(f"   1. Check your API key at: https://openrouter.ai/settings/keys")
                        print(f"   2. Add credits or update to a new API key")
                        print(f"   3. Update OPENROUTER_API_KEY in this script (line ~56)")
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
                        print(f"\n❌ Error evaluating {model} on {dataset_name}: {e}")
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
    
    # Print results
    print_results_table(all_results)
    print_idk_rate_table(all_results)
    print_comparison_tables(all_comparisons)
    
    # Print timing summary
    print(f"\n{'='*80}")
    print("OVERALL TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Start time:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_time:.2f}s ({total_time/60:.2f} min / {total_time/3600:.2f} hours)")
    print(f"Total questions processed: {overall_question_idx['idx']}")
    print(f"Avg per question (wall time): {total_time/overall_question_idx['idx']:.2f}s" if overall_question_idx['idx'] > 0 else "N/A")
    print(f"{'='*80}")
    
    # Save results
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'q4_results': all_results,
                'q2_comparison': all_comparisons
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Final results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"\n⚠ Error saving results: {e}")
    
    print(f"\n{'='*80}")
    print("Q4 ASYNC EVALUATION COMPLETED!")
    print(f"{'='*80}")


def main():
    """Wrapper to run async main."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
