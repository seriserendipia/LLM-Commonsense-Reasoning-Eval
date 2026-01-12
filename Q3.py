"""
Q3: LLM Robustness Evaluation - Random Option Removal Experiment

This script evaluates the robustness of three LLMs by testing them on CommonsenseQA
with a modified format:
- Original: 5 options (4 wrong + 1 correct)
- Modified: 4 options (3 wrong + 1 correct) - randomly remove 1 wrong option
- Options are randomly shuffled to eliminate position bias

The goal is to assess whether removing one wrong option affects LLM accuracy.

Output: 3x1 accuracy table (3 models × CommonsenseQA only)
"""

import os
import json
import re
import time
import random
from datetime import datetime, timedelta
from openai import OpenAI
from typing import List, Dict, Tuple
from tqdm import tqdm

# Import data loading functions from load_data.py
from load_data import loaded_datasets
from api_key_loader import load_openrouter_api_key

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

# Only evaluate CommonsenseQA for Q3
DATASETS = ["CommonsenseQA"]

# Evaluation parameters
MAX_SAMPLES = 0  # Set to 0 to evaluate all samples, or N to evaluate first N samples
REQUEST_DELAY = 0.0  # Delay between API requests (seconds) to avoid rate limiting
MAX_RETRIES = 3  # Maximum number of retries for failed requests
MAX_TOKENS = 20  # Maximum tokens to generate in LLM response

# Cache settings
USE_CACHE = True  # Set to False to ignore cache and re-evaluate everything
CACHE_FILE = "Q3_evaluation_cache.json"  # Q3 specific cache file
RESULTS_FILE = "Q3_final_results.json"  # Q3 specific results file

# Random seed for reproducibility
RANDOM_SEED_BASE = 42  # Base seed, each sample will use (RANDOM_SEED_BASE + sample_index)

# ============================================================================
# Initialize OpenAI Client for OpenRouter
# ============================================================================

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY
)

print("="*80)
print("Q3: LLM ROBUSTNESS EVALUATION - RANDOM OPTION REMOVAL")
print("="*80)
print(f"\nExperiment Design:")
print(f"  Original format: 5 options (4 wrong + 1 correct)")
print(f"  Modified format: 4 options (3 wrong + 1 correct)")
print(f"  Strategy: Randomly remove 1 wrong option, then shuffle all options")
print(f"  Random seed: Based on sample index (reproducible)")
print(f"\nConfiguration:")
print(f"  Models: {len(MODELS)}")
for model in MODELS:
    print(f"    - {model}")
print(f"  Dataset: CommonsenseQA only")
print(f"  Max samples: {'All' if MAX_SAMPLES == 0 else MAX_SAMPLES}")
print(f"  Request delay: {REQUEST_DELAY}s")
print(f"  Use cache: {USE_CACHE}")
print(f"  Cache file: {CACHE_FILE}")

# ============================================================================
# Q3 Specific: Random Option Removal and Shuffling
# ============================================================================

def process_sample_with_random_drop(sample, sample_index):
    """
    Process a CommonsenseQA sample by randomly removing one wrong option
    and shuffling the remaining 4 options.
    
    Args:
        sample (dict): Original sample with 5 options
            - question: str
            - choices: list of 5 strings
            - correct_answer: int (0-4, index of correct answer)
        sample_index (int): Index of the sample (used as random seed)
    
    Returns:
        dict: Modified sample with 4 shuffled options
            - question: str (unchanged)
            - choices: list of 4 strings (shuffled)
            - correct_answer: int (0-3, new index of correct answer)
            - original_correct_index: int (for debugging)
            - removed_choice_index: int (which wrong option was removed)
    """
    # Extract original data
    original_choices = sample['choices']  # List of 5 choice texts
    correct_index = sample['correct_answer']  # Index of correct answer (0-4)
    
    # Step 1: Identify all wrong answer indices
    all_indices = list(range(len(original_choices)))  # [0, 1, 2, 3, 4]
    wrong_indices = [i for i in all_indices if i != correct_index]  # 4 wrong indices
    
    # Step 2: Create a deterministic random generator using sample index as seed
    # This ensures the same sample always gets the same random result
    rng = random.Random(RANDOM_SEED_BASE + sample_index)
    
    # Step 3: Randomly select 3 wrong options to keep (remove 1 wrong option)
    selected_wrong_indices = rng.sample(wrong_indices, 3)
    removed_index = [i for i in wrong_indices if i not in selected_wrong_indices][0]
    
    # Step 4: Combine selected wrong options with the correct option
    new_indices = selected_wrong_indices + [correct_index]  # 4 indices total
    
    # Step 5: Shuffle the 4 options to eliminate position bias
    rng.shuffle(new_indices)
    
    # Step 6: Build new choices list based on shuffled indices
    new_choices = [original_choices[i] for i in new_indices]
    
    # Step 7: Find the new index of the correct answer in the shuffled list
    new_correct_index = new_indices.index(correct_index)
    
    # Return modified sample
    return {
        'question': sample['question'],
        'choices': new_choices,  # 4 shuffled options
        'correct_answer': new_correct_index,  # New index (0-3)
        'original_correct_index': correct_index,  # Original index (0-4) for debugging
        'removed_choice_index': removed_index  # Which option was removed
    }


def format_commonsenseqa_sample_q3(sample):
    """
    Convert CommonsenseQA sample to unified format (original 5 options).
    This is the same as the Q2 version before modification.
    
    Original format:
        - question: str (the question text)
        - choices: dict with 'label' and 'text' lists
        - answerKey: str (e.g., 'A', 'B', 'C', 'D', 'E')
    
    Unified format:
        - question: str
        - choices: list of str (choice texts)
        - correct_answer: int (index of correct answer, 0-based)
    
    Args:
        sample (dict): Original CommonsenseQA sample
    
    Returns:
        dict: Unified format sample
    """
    return {
        'question': sample['question'],
        'choices': sample['choices']['text'],
        'correct_answer': sample['choices']['label'].index(sample['answerKey'])
    }


def get_formatted_validation_data_q3(max_samples=0, print_examples=True):
    """
    Get validation data for CommonsenseQA with random option removal.
    
    This function:
    1. Loads CommonsenseQA validation split
    2. Converts to unified format (5 options)
    3. Applies random option removal and shuffling (4 options)
    4. Returns list of modified samples
    
    Args:
        max_samples (int): Maximum number of samples to return.
            If 0, return all samples. Default: 0
        print_examples (bool): Whether to print example transformations
    
    Returns:
        list of dict: List of modified samples with 4 shuffled options
    """
    dataset_name = 'CommonsenseQA'
    formatted_samples = []
    
    print(f"\n{'='*80}")
    print(f"Loading and Processing {dataset_name} Validation Data")
    print(f"{'='*80}")
    
    if dataset_name not in loaded_datasets:
        print(f"❌ Error: {dataset_name} not found in loaded datasets")
        return []
    
    dataset = loaded_datasets[dataset_name]
    
    if 'validation' not in dataset:
        print(f"❌ Error: validation split not found in {dataset_name}")
        return []
    
    validation_data = dataset['validation']
    total_samples = len(validation_data)
    
    print(f"✓ Found {total_samples} samples in validation split")
    
    # Determine how many samples to process
    num_samples = total_samples if max_samples == 0 else min(max_samples, total_samples)
    print(f"✓ Processing {num_samples} samples")
    
    # Process each sample
    for idx in range(num_samples):
        original_sample = validation_data[idx]
        
        # Convert to unified format (5 options)
        unified_sample = format_commonsenseqa_sample_q3(original_sample)
        
        # Apply random drop and shuffle (4 options)
        modified_sample = process_sample_with_random_drop(unified_sample, idx)
        
        formatted_samples.append(modified_sample)
    
    print(f"✓ Successfully processed {len(formatted_samples)} samples")
    
    # Print examples of the transformation
    if print_examples and len(formatted_samples) > 0:
        print(f"\n{'='*80}")
        print("EXAMPLE TRANSFORMATIONS (First 2 samples)")
        print(f"{'='*80}")
        
        num_examples = min(2, len(formatted_samples))
        
        for i in range(num_examples):
            sample = formatted_samples[i]
            original_sample = validation_data[i]
            
            print(f"\n{'─'*80}")
            print(f"Sample #{i}")
            print(f"{'─'*80}")
            print(f"Question: {sample['question'][:100]}...")
            
            # Show original 5 options
            print(f"\nOriginal (5 options):")
            original_labels = original_sample['choices']['label']
            original_texts = original_sample['choices']['text']
            original_answer = original_sample['answerKey']
            
            for label, text in zip(original_labels, original_texts):
                marker = "✓" if label == original_answer else " "
                print(f"  [{marker}] {label}. {text[:50]}...")
            
            # Show which option was removed
            removed_idx = sample['removed_choice_index']
            removed_text = original_texts[removed_idx]
            print(f"\n❌ Removed option: {chr(65 + removed_idx)}. {removed_text[:50]}...")
            
            # Show modified 4 options (shuffled)
            print(f"\nModified (4 shuffled options):")
            new_labels = ['A', 'B', 'C', 'D']
            for j, (label, text) in enumerate(zip(new_labels, sample['choices'])):
                marker = "✓" if j == sample['correct_answer'] else " "
                print(f"  [{marker}] {label}. {text[:50]}...")
            
            print(f"\nCorrect answer: {new_labels[sample['correct_answer']]}")
    
    return formatted_samples


# ============================================================================
# Helper Functions (Copied from Q2.py)
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


def build_prompt(question: str, choices: List[str]) -> str:
    """
    Build a prompt for the LLM to answer a multiple-choice question.
    This function works with any number of choices (automatically adapts to 4 choices).
    
    Args:
        question (str): The question text
        choices (List[str]): List of answer choices (4 choices for Q3)
    
    Returns:
        str: Formatted prompt
    """
    # Create choice labels (A, B, C, D for 4 choices)
    choice_labels = [chr(65 + i) for i in range(len(choices))]  # A, B, C, D
    
    # Format choices
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    # Build prompt
    prompt = f"""Question: {question}

Choices:
{choices_text}

Please select the best answer from the choices above. Reply with ONLY the letter (A, B, C, etc.) of your chosen answer. Do not include any explanation or additional text."""
    
    return prompt


def call_llm(model: str, prompt: str, max_retries: int = MAX_RETRIES) -> tuple:
    """
    Call the LLM via OpenRouter API with retry logic.
    
    Args:
        model (str): Model identifier
        prompt (str): Input prompt
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        tuple: (response_text, was_truncated) where:
            - response_text: str, LLM response text or None if failed
            - was_truncated: bool, True if response was truncated due to max_tokens
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0  # Deterministic responses
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            was_truncated = (finish_reason == 'length')
            
            if was_truncated:
                print(f"  ⚠ Response truncated (finish_reason: {finish_reason})")
            
            return (response_text, was_truncated)
            
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"  ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ All {max_retries} attempts failed")
    
    return (None, False)


def parse_answer(llm_response: str, num_choices: int) -> int:
    """
    Parse the LLM's response to extract the answer choice index.
    Works with any number of choices (4 choices for Q3).
    
    Args:
        llm_response (str): Raw response from LLM
        num_choices (int): Number of available choices (4 for Q3)
    
    Returns:
        int: Answer index (0-based), or -1 if parsing failed
    """
    if llm_response is None:
        return -1
    
    # Clean up response
    response = llm_response.strip().upper()
    
    letter = None
    
    # Try to find a single letter (A, B, C, D)
    if len(response) == 1 and response.isalpha():
        letter = response
    else:
        # Pattern: Letter followed by punctuation (e.g., "A.", "A)", "A:")
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
    
    # Convert letter to index (A=0, B=1, C=2, D=3)
    answer_index = ord(letter) - ord('A')
    
    # Validate index is within valid range
    if 0 <= answer_index < num_choices:
        return answer_index
    else:
        return -1


def evaluate_model_on_dataset(model: str, dataset_name: str, samples: List[Dict], cache: Dict, 
                             progress_bar=None, overall_question_idx=None) -> Dict:
    """
    Evaluate a single model on a single dataset with timing and progress tracking.
    
    Args:
        model (str): Model identifier
        dataset_name (str): Dataset name
        samples (List[Dict]): List of samples in unified format
        cache (Dict): Cache dictionary for storing results
        progress_bar: Optional tqdm progress bar for overall progress
        overall_question_idx: Optional dict to track overall question index
    
    Returns:
        Dict: Evaluation results containing accuracy, detailed stats, and timing info
    """
    cache_key = f"{model}::{dataset_name}"
    
    # Check if already evaluated (only if USE_CACHE is True)
    if USE_CACHE and cache_key in cache and len(cache[cache_key].get('results', [])) == len(samples):
        print(f"  ✓ Using cached results for {model} on {dataset_name}")
        if progress_bar and overall_question_idx is not None:
            overall_question_idx['idx'] += len(samples)
            progress_bar.update(len(samples))
        return cache[cache_key]
    
    if not USE_CACHE and cache_key in cache:
        print(f"  ⚠ Cache exists but ignored (USE_CACHE=False), re-evaluating...")
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model} on {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print()
    
    # Initialize results and timing
    results = []
    correct_count = 0
    failed_count = 0
    length_limited_count = 0
    question_times = []
    
    dataset_start_time = time.time()
    
    # Evaluate each sample
    for idx, sample in enumerate(samples):
        question_start_time = time.time()
        
        # Build prompt
        prompt = build_prompt(sample['question'], sample['choices'])
        
        # Call LLM
        llm_response, was_truncated = call_llm(model, prompt)
        
        # Parse answer
        predicted_index = parse_answer(llm_response, len(sample['choices']))
        
        # Check correctness
        is_correct = (predicted_index == sample['correct_answer'])
        
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
        avg_time = sum(question_times) / len(question_times)
        if progress_bar and overall_question_idx is not None:
            overall_question_idx['idx'] += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                'model': model.split('/')[-1][:15],
                'dataset': dataset_name[:10],
                'avg_time': f'{avg_time:.2f}s'
            })
        
        # Store result
        result = {
            'sample_index': idx,
            'question': sample['question'][:100],
            'correct_answer': sample['correct_answer'],
            'predicted_answer': predicted_index,
            'llm_response': llm_response,
            'is_correct': is_correct,
            'parse_failed': (predicted_index == -1),
            'length_limited': was_truncated,
            'time_seconds': question_time
        }
        results.append(result)
        
        # Add delay to avoid rate limiting (except for last request)
        if idx < len(samples) - 1:
            time.sleep(REQUEST_DELAY)
    
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
    print(f"Evaluation Summary: {model} on {dataset_name}")
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
    
    # Store in cache
    evaluation_result = {
        'model': model,
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


def print_results_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print the final 3x1 results table.
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """
    print(f"\n{'='*80}")
    print("Q3 FINAL RESULTS: ACCURACY TABLE (%) - 4 Options (3 Wrong + 1 Correct)")
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
            if dataset in all_results.get(model, {}):
                accuracy = all_results[model][dataset]['accuracy']
                row += f"{accuracy:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print("-" * 80)


def print_detailed_statistics_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print a detailed statistics table with accuracy, length-limited rate, and wrong answer rate.
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """

    print(f"\n{'='*120}")
    print(f"Q3 DETAILED STATISTICS: ACCURACY, LENGTH-LIMITED RATE & WRONG ANSWER RATE")
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
            if dataset in all_results.get(model, {}):
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
            if dataset in all_results.get(model, {}):
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
            if dataset in all_results.get(model, {}):
                result = all_results[model][dataset]
                total = result['total_samples']
                failed = result['failed_count']
                # Wrong answers (mutually exclusive: correct + wrong + length_limited = 100%)
                wrong_rate = (failed / total * 100) if total > 0 else 0.0
                row += f"{wrong_rate:>14.2f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 120)


def print_timing_statistics_table(all_results: Dict[str, Dict[str, Dict]]):
    """
    Print a timing statistics table.
    
    Args:
        all_results (Dict): Nested dict: model -> dataset -> results
    """
    print(f"\n{'='*120}")
    print(f"Q3 TIMING STATISTICS")
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
            if dataset in all_results.get(model, {}):
                time_val = all_results[model][dataset]['total_time_seconds']
                row += f"{time_val:>14.2f}s"
                model_total += time_val
            else:
                row += f"{'N/A':>15}"
        model_totals[model] = model_total
        row += f"{model_total:>17.2f}s"
        print(row)
    
    print("=" * 120)


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def main():
    """Main evaluation function with overall timing."""
    overall_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("STARTING Q3 EVALUATION")
    print(f"{'='*80}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS)}")
    print(f"Dataset: CommonsenseQA only")
    print(f"Total combinations: {len(MODELS)} × 1 = {len(MODELS)}")
    
    # Load cache
    cache = load_cache()
    
    # Load and process CommonsenseQA data with random option removal
    samples = get_formatted_validation_data_q3(max_samples=MAX_SAMPLES, print_examples=True)
    
    if len(samples) == 0:
        print("\n❌ No samples loaded. Exiting.")
        return
    
    # Store all results: model -> dataset -> results
    all_results = {}
    
    total_questions = len(samples) * len(MODELS)
    print(f"\nTotal questions to process: {total_questions} ({len(samples)} questions × {len(MODELS)} models)")
    
    # Create overall progress bar
    print(f"\n{'='*80}")
    print("EVALUATION PROGRESS")
    print(f"{'='*80}\n")
    
    overall_pbar = tqdm(
        total=total_questions,
        desc="Overall Progress",
        unit="q",
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    )
    
    overall_question_idx = {'idx': 0}
    
    # Evaluate each model on CommonsenseQA
    for model in MODELS:
        all_results[model] = {}
        
        # Evaluate this model
        result = evaluate_model_on_dataset(
            model=model,
            dataset_name='CommonsenseQA',
            samples=samples,
            cache=cache,
            progress_bar=overall_pbar,
            overall_question_idx=overall_question_idx
        )
        
        all_results[model]['CommonsenseQA'] = result
    
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
    print(f"Avg per question: {total_time/overall_question_idx['idx']:.2f}s" if overall_question_idx['idx'] > 0 else "N/A")
    print(f"{'='*80}")
    
    # Save final results to a separate file
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Final results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"\n⚠ Error saving final results: {e}")
    
    print(f"\n{'='*80}")
    print("Q3 EVALUATION COMPLETED!")
    print(f"{'='*80}")
    print(f"\n💡 Compare these results with Q2 (5 options) to assess robustness.")
    print(f"   Expected: Accuracy may increase slightly (random guess: 25% vs 20%)")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
