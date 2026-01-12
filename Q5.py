"""
Q5: Question Paraphrasing Robustness Evaluation

This script evaluates LLM robustness to question paraphrasing:
1. Takes 50 questions from PIQA validation split
2. Uses qwen/qwen-2.5-72b-instruct to paraphrase questions (preserve meaning, change wording)
3. Keeps options/answers unchanged
4. Evaluates all 3 LLMs on paraphrased questions
5. Compares results with Q2 to measure consistency

Analysis:
- Overall consistency: How many questions have same answer as Q2?
- Q2 correct → Q5 wrong: Questions where LLM was right in Q2 but wrong here
- Detailed table with original/paraphrased questions, Q2/Q5 answers
"""

import os
import sys
import json
import asyncio
import warnings
import logging
import time
from datetime import datetime
from openai import AsyncOpenAI, RateLimitError
from typing import List, Dict, Tuple
from tqdm import tqdm
import re

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

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

# Paraphrasing model
PARAPHRASE_MODEL = "qwen/qwen-2.5-72b-instruct"

# Dataset and sample configuration
DATASET_NAME = "PIQA"
NUM_SAMPLES = 50  # Take first 50 questions from validation split
MAX_TOKENS = 20  # Maximum tokens for answer generation

# Async & Retry Configuration
MAX_CONCURRENT_REQUESTS = 50  # Lower for paraphrasing to avoid rate limits
MAX_RETRIES = 3

# File paths
PARAPHRASED_FILE = "Q5_paraphrased_questions.json"
Q2_RESULTS_FILE = "Q2_async_final_results.json"
Q5_RESULTS_FILE = "Q5_final_results.json"

# Paraphrasing control
FORCE_REPARAPHRASE = False  # Set to True to force regenerating paraphrased questions, ignoring cache

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
print("Q5: QUESTION PARAPHRASING ROBUSTNESS EVALUATION")
print("="*80)
print(f"\nExperiment Design:")
print(f"  1. Take {NUM_SAMPLES} questions from {DATASET_NAME} validation split")
print(f"  2. Use {PARAPHRASE_MODEL} to paraphrase questions")
print(f"  3. Keep options/answers unchanged")
print(f"  4. Evaluate all {len(MODELS)} LLMs on paraphrased questions")
print(f"  5. Compare with Q2 results for consistency analysis")
print(f"\nConfiguration:")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Number of questions: {NUM_SAMPLES}")
print(f"  Paraphrase model: {PARAPHRASE_MODEL}")
print(f"  Evaluation models: {len(MODELS)}")
for model in MODELS:
    print(f"    - {model}")
print(f"  Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
print(f"  Paraphrased questions file: {PARAPHRASED_FILE}")
print(f"  Q2 results file: {Q2_RESULTS_FILE}")
print(f"  Force reparaphrase: {FORCE_REPARAPHRASE}")
print("="*80)

# ============================================================================
# Load Q2 Results
# ============================================================================

def load_q2_results():
    """Load Q2 results for PIQA dataset."""
    if not os.path.exists(Q2_RESULTS_FILE):
        print(f"\n❌ ERROR: Q2 results file not found: {Q2_RESULTS_FILE}")
        print(f"   Please run Q2_async.py first to generate Q2 results.")
        sys.exit(1)
    
    try:
        with open(Q2_RESULTS_FILE, 'r', encoding='utf-8') as f:
            q2_results = json.load(f)
        
        # Extract PIQA results for each model
        piqa_results = {}
        for model in MODELS:
            if model in q2_results and DATASET_NAME in q2_results[model]:
                piqa_results[model] = q2_results[model][DATASET_NAME]
            else:
                print(f"\n❌ ERROR: Q2 results not found for {model} on {DATASET_NAME}")
                sys.exit(1)
        
        print(f"\n✓ Loaded Q2 results from {Q2_RESULTS_FILE}")
        print(f"  PIQA samples in Q2: {piqa_results[MODELS[0]]['total_samples']}")
        
        return piqa_results
    
    except Exception as e:
        print(f"\n❌ Error loading Q2 results: {e}")
        sys.exit(1)


# ============================================================================
# Paraphrasing Functions
# ============================================================================

def build_paraphrase_prompt(question: str, choices: List[str]) -> str:
    """
    Build prompt for paraphrasing a question.
    
    Args:
        question: Original question text
        choices: List of answer choices (to show context)
    
    Returns:
        Prompt for paraphrasing
    """
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""Please paraphrase the following question. Preserve the exact meaning but change the wording. 

IMPORTANT: Your paraphrased question MUST be different from the original question. Do NOT just copy the original question. Use different words, sentence structure, or phrasing while keeping the same meaning.

Original Question: {question}

Answer Choices (for context only, do NOT include in your response):
{choices_text}

Paraphrased Question (question text only, no choices):"""
    
    return prompt


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def paraphrase_question_async(client: AsyncOpenAI, question: str, choices: List[str]) -> str:
    """
    Use LLM to paraphrase a question asynchronously.
    
    Args:
        client: AsyncOpenAI client
        question: Original question text
        choices: List of answer choices (for context)
    
    Returns:
        Paraphrased question text
    """
    prompt = build_paraphrase_prompt(question, choices)
    
    try:
        print(f"\n[DEBUG] 🔍 Sending paraphrase request...")
        print(f"[DEBUG] Original Q: {question[:80]}...")
        
        response = await client.chat.completions.create(
            model=PARAPHRASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,  # Longer for paraphrasing
            temperature=0.7  # Some creativity for paraphrasing
        )
        
        # Check response validity
        if not response or not response.choices:
            print(f"[DEBUG] ❌ Empty response from API!")
            return question
        
        paraphrased = response.choices[0].message.content.strip()
        
        print(f"[DEBUG] Raw response: {paraphrased[:80]}...")
        
        # Remove common prefixes if present
        prefixes = ["Paraphrased Question:", "Question:", "Paraphrased:"]
        for prefix in prefixes:
            if paraphrased.startswith(prefix):
                paraphrased = paraphrased[len(prefix):].strip()
                print(f"[DEBUG] Removed prefix: {prefix}")
        
        # Check if paraphrased is same as original
        if paraphrased == question:
            print(f"[DEBUG] ⚠️  WARNING: Paraphrased is IDENTICAL to original!")
        elif paraphrased.lower() == question.lower():
            print(f"[DEBUG] ⚠️  WARNING: Paraphrased is same (case-insensitive)!")
        else:
            print(f"[DEBUG] ✓ Paraphrased is different from original")
        
        print(f"[DEBUG] Final paraphrased: {paraphrased[:80]}...")
        
        return paraphrased
        
    except RateLimitError as e:
        print(f"[DEBUG] ⚠️  Rate limit hit, Tenacity will retry...")
        logger.warning(f"Rate limit hit during paraphrasing, retrying...")
        raise
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ EXCEPTION OCCURRED DURING PARAPHRASING")
        print(f"{'='*80}")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {e}")
        print(f"Original Question: {question}")
        print(f"{'='*80}\n")
        logger.error(f"Error paraphrasing question: {e}")
        # Re-raise the exception instead of returning original question
        raise


async def paraphrase_all_questions(samples: List[Dict]) -> List[Dict]:
    """
    Paraphrase all questions asynchronously.
    
    Args:
        samples: List of original samples with questions
    
    Returns:
        List of samples with paraphrased questions
    """
    print(f"\n{'='*80}")
    print(f"PARAPHRASING {len(samples)} QUESTIONS")
    print(f"{'='*80}")
    print(f"Using model: {PARAPHRASE_MODEL}")
    print(f"This may take a few minutes...")
    print()
    
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def paraphrase_one(sample, idx):
        async with semaphore:
            original_question = sample['question']
            choices = sample['choices']
            
            paraphrased_question = await paraphrase_question_async(client, original_question, choices)
            
            return {
                'index': idx,
                'original_question': original_question,
                'paraphrased_question': paraphrased_question,
                'choices': choices,
                'correct_answer': sample['correct_answer']
            }
    
    # Create progress bar
    pbar = tqdm(total=len(samples), desc="Paraphrasing", unit="q")
    
    # Paraphrase all questions
    tasks = [paraphrase_one(sample, idx) for idx, sample in enumerate(samples)]
    paraphrased_samples = []
    
    for task in asyncio.as_completed(tasks):
        result = await task
        paraphrased_samples.append(result)
        pbar.update(1)
    
    pbar.close()
    
    # Sort by index to maintain order
    paraphrased_samples.sort(key=lambda x: x['index'])
    
    print(f"\n✓ Paraphrasing complete!")
    print(f"\nExample paraphrases:")
    for i in range(min(3, len(paraphrased_samples))):
        sample = paraphrased_samples[i]
        print(f"\n  Question {i+1}:")
        print(f"    Original:    {sample['original_question'][:100]}...")
        print(f"    Paraphrased: {sample['paraphrased_question'][:100]}...")
    
    return paraphrased_samples


def save_paraphrased_questions(paraphrased_samples: List[Dict]):
    """Save paraphrased questions to JSON file."""
    try:
        with open(PARAPHRASED_FILE, 'w', encoding='utf-8') as f:
            json.dump(paraphrased_samples, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Paraphrased questions saved to {PARAPHRASED_FILE}")
    except Exception as e:
        print(f"\n⚠ Error saving paraphrased questions: {e}")


def load_paraphrased_questions() -> List[Dict]:
    """Load paraphrased questions from JSON file if exists."""
    if os.path.exists(PARAPHRASED_FILE):
        try:
            with open(PARAPHRASED_FILE, 'r', encoding='utf-8') as f:
                paraphrased_samples = json.load(f)
            print(f"\n✓ Loaded {len(paraphrased_samples)} paraphrased questions from {PARAPHRASED_FILE}")
            return paraphrased_samples
        except Exception as e:
            print(f"\n⚠ Error loading paraphrased questions: {e}")
            return None
    return None


# ============================================================================
# Evaluation Functions (Reused from Q4)
# ============================================================================

def build_prompt(question: str, choices: List[str]) -> str:
    """Build prompt for LLM to answer a multiple-choice question."""
    choice_labels = [chr(65 + i) for i in range(len(choices))]
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    prompt = f"""Question: {question}

Choices:
{choices_text}

Please select the best answer from the choices above. Reply with ONLY the letter (A, B, C, etc.) of your chosen answer. Do not include any explanation or additional text."""
    
    return prompt


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def call_llm_async(client: AsyncOpenAI, model: str, prompt: str) -> Tuple[str, bool]:
    """Call LLM asynchronously with retry logic."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )
        
        message_content = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        was_truncated = (finish_reason == "length")
        
        return (message_content, was_truncated)
        
    except RateLimitError as e:
        logger.warning(f"Rate limit hit for model {model}, retrying...")
        raise
    except Exception as e:
        logger.error(f"Error calling {model}: {e}")
        return (None, False)


def parse_answer(llm_response: str, num_choices: int, was_truncated: bool = False) -> int:
    """Parse LLM response to extract answer choice index."""
    if llm_response is None:
        return -1
    
    response = llm_response.strip().upper()
    
    # Check if response was truncated with thinking keywords
    if was_truncated:
        thinking_keywords = [
            "WE NEED TO", "LET'S THINK", "LET ME ANALYZE",
            "FIRST, WE SHOULD", "TO SOLVE THIS", "LET'S BREAK DOWN"
        ]
        if any(keyword in response for keyword in thinking_keywords):
            return -1
    
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
            match = re.search(r'([A-Z])', response)
            if match:
                letter = match.group(1)
    
    if letter is None:
        return -1
    
    answer_index = ord(letter) - ord('A')
    
    if 0 <= answer_index < num_choices:
        return answer_index
    else:
        return -1


async def evaluate_model_on_paraphrased_questions(
    model_name: str,
    paraphrased_samples: List[Dict],
    progress_bar=None
) -> Dict:
    """
    Evaluate a single model on paraphrased questions.
    
    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} on Paraphrased {DATASET_NAME}")
    print(f"{'='*80}")
    print(f"Total samples: {len(paraphrased_samples)}")
    print()
    
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    results = []
    correct_count = 0
    failed_count = 0
    question_times = []
    
    start_time = time.time()
    
    async def process_one_question(sample, idx):
        nonlocal correct_count, failed_count
        
        question_start_time = time.time()
        
        async with semaphore:
            # Use paraphrased question
            question = sample['paraphrased_question']
            choices = sample['choices']
            correct_answer = sample['correct_answer']
            
            # Build prompt and call LLM
            prompt = build_prompt(question, choices)
            llm_response, was_truncated = await call_llm_async(client, model_name, prompt)
            
            # Parse answer
            predicted_answer = parse_answer(llm_response, len(choices), was_truncated)
            
            # Check correctness
            is_correct = (predicted_answer == correct_answer)
            
            if predicted_answer == -1:
                failed_count += 1
            elif is_correct:
                correct_count += 1
            
            question_time = time.time() - question_start_time
            question_times.append(question_time)
            
            if progress_bar:
                progress_bar.update(1)
            
            return {
                'index': idx,
                'original_question': sample['original_question'],
                'paraphrased_question': sample['paraphrased_question'],
                'choices': choices,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'llm_response': llm_response,
                'is_correct': is_correct,
                'was_truncated': was_truncated,
                'time_seconds': question_time
            }
    
    # Execute all tasks
    tasks = [process_one_question(sample, idx) for idx, sample in enumerate(paraphrased_samples)]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_time = sum(question_times) / len(question_times) if question_times else 0.0
    
    total = len(paraphrased_samples)
    accuracy = (correct_count / total * 100) if total > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"Evaluation Summary: {model_name}")
    print(f"{'='*80}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {total - correct_count - failed_count}")
    print(f"  Parse failed: {failed_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg per question: {avg_time:.2f}s")
    
    return {
        'model': model_name,
        'total_samples': total,
        'correct_count': correct_count,
        'failed_count': failed_count,
        'accuracy': accuracy,
        'total_time_seconds': total_time,
        'avg_time_per_question': avg_time,
        'results': results
    }


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_q2_q5_results(q5_results: Dict, q2_results: Dict):
    """
    Compare Q5 results with Q2 results for each model.
    
    Returns:
        Dict with comparison statistics for each model
    """
    comparisons = {}
    
    for model in MODELS:
        q5_model_results = q5_results[model]['results']
        q2_model_results = q2_results[model]['results'][:NUM_SAMPLES]  # First 50 from Q2
        
        # Ensure same number of samples
        if len(q5_model_results) != len(q2_model_results):
            print(f"\n⚠ Warning: Sample count mismatch for {model}")
            print(f"   Q5: {len(q5_model_results)}, Q2: {len(q2_model_results)}")
            continue
        
        # Statistics
        same_answer = 0
        q2_correct_q5_wrong = []
        q2_wrong_q5_correct = []
        both_correct = 0
        both_wrong = 0
        
        for idx in range(len(q5_model_results)):
            q5_sample = q5_model_results[idx]
            q2_sample = q2_model_results[idx]
            
            q5_pred = q5_sample['predicted_answer']
            q2_pred = q2_sample['predicted_answer']
            
            q5_correct = q5_sample['is_correct']
            q2_correct = q2_sample['is_correct']
            
            # Same answer (regardless of correctness)
            if q5_pred == q2_pred:
                same_answer += 1
            
            # Correctness transitions
            if q2_correct and q5_correct:
                both_correct += 1
            elif q2_correct and not q5_correct:
                q2_correct_q5_wrong.append({
                    'index': idx,
                    'original_question': q2_sample.get('question', 'N/A'),
                    'paraphrased_question': q5_sample['paraphrased_question'],
                    'choices': q5_sample['choices'],
                    'correct_answer': q5_sample['correct_answer'],
                    'correct_answer_letter': chr(65 + q5_sample['correct_answer']),
                    'q2_answer': q2_pred,
                    'q2_answer_letter': chr(65 + q2_pred) if q2_pred != -1 else 'Failed',
                    'q5_answer': q5_pred,
                    'q5_answer_letter': chr(65 + q5_pred) if q5_pred != -1 else 'Failed'
                })
            elif not q2_correct and q5_correct:
                q2_wrong_q5_correct.append(idx)
            else:
                both_wrong += 1
        
        total = len(q5_model_results)
        consistency_rate = (same_answer / total * 100) if total > 0 else 0.0
        
        comparisons[model] = {
            'total_samples': total,
            'same_answer': same_answer,
            'consistency_rate': consistency_rate,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'q2_correct_q5_wrong': q2_correct_q5_wrong,
            'q2_wrong_q5_correct': q2_wrong_q5_correct,
            'q2_accuracy': (sum(1 for s in q2_model_results if s['is_correct']) / total * 100),
            'q5_accuracy': (sum(1 for s in q5_model_results if s['is_correct']) / total * 100)
        }
    
    return comparisons


def print_comparison_tables(comparisons: Dict):
    """Print comparison tables and detailed analysis."""
    
    # Table 1: Accuracy Comparison
    print(f"\n{'='*80}")
    print("ACCURACY COMPARISON: Q2 (Original) vs Q5 (Paraphrased)")
    print(f"{'='*80}")
    print(f"{'Model':<40}{'Q2 Acc %':>12}{'Q5 Acc %':>12}{'Diff %':>12}")
    print("-" * 80)
    
    for model in MODELS:
        if model in comparisons:
            comp = comparisons[model]
            q2_acc = comp['q2_accuracy']
            q5_acc = comp['q5_accuracy']
            diff = q5_acc - q2_acc
            print(f"{model:<40}{q2_acc:>11.2f}{q5_acc:>11.2f}{diff:>+11.2f}")
    
    print("-" * 80)
    
    # Table 2: Consistency Analysis
    print(f"\n{'='*80}")
    print("CONSISTENCY ANALYSIS: Same Answer Rate")
    print(f"{'='*80}")
    print(f"{'Model':<40}{'Same Answer':>15}{'Rate %':>12}")
    print("-" * 80)
    
    for model in MODELS:
        if model in comparisons:
            comp = comparisons[model]
            same = comp['same_answer']
            total = comp['total_samples']
            rate = comp['consistency_rate']
            print(f"{model:<40}{same}/{total:>3}{rate:>11.2f}")
    
    print("-" * 80)
    
    # Table 3: Correctness Transitions
    print(f"\n{'='*80}")
    print("CORRECTNESS TRANSITIONS: Q2 → Q5")
    print(f"{'='*80}")
    print(f"{'Model':<30}{'Both Correct':>14}{'Q2✓→Q5✗':>12}{'Q2✗→Q5✓':>12}{'Both Wrong':>12}")
    print("-" * 80)
    
    for model in MODELS:
        if model in comparisons:
            comp = comparisons[model]
            print(f"{model:<30}"
                  f"{comp['both_correct']:>14}"
                  f"{len(comp['q2_correct_q5_wrong']):>12}"
                  f"{len(comp['q2_wrong_q5_correct']):>12}"
                  f"{comp['both_wrong']:>12}")
    
    print("-" * 80)
    
    # Detailed table for Q2 correct → Q5 wrong
    print(f"\n{'='*80}")
    print("DETAILED: Questions where LLM was CORRECT in Q2 but WRONG in Q5")
    print(f"{'='*80}")
    
    for model in MODELS:
        if model not in comparisons:
            continue
        
        errors = comparisons[model]['q2_correct_q5_wrong']
        
        if not errors:
            print(f"\n{model}: No such cases (100% maintained correctness)")
            continue
        
        print(f"\n{model}: {len(errors)} case(s)")
        print("-" * 80)
        
        for error in errors:
            idx = error['index']
            print(f"\nQuestion #{idx + 1}:")
            print(f"  Original Question:")
            print(f"    {error['original_question']}")
            print(f"  Paraphrased Question:")
            print(f"    {error['paraphrased_question']}")
            print(f"  Choices:")
            for i, choice in enumerate(error['choices']):
                print(f"    {chr(65+i)}. {choice}")
            print(f"  Correct Answer: {error['correct_answer_letter']}")
            print(f"  Q2 Answer: {error['q2_answer_letter']} (Correct)")
            print(f"  Q5 Answer: {error['q5_answer_letter']} (Wrong)")
            print()


# ============================================================================
# Main Evaluation Loop
# ============================================================================

async def main_async():
    """Main async evaluation function."""
    overall_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("STARTING Q5 EVALUATION")
    print(f"{'='*80}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load Q2 results
    print(f"\n{'='*80}")
    print("STEP 1: Load Q2 Results")
    print(f"{'='*80}")
    q2_results = load_q2_results()
    
    # Step 2: Load or create paraphrased questions
    print(f"\n{'='*80}")
    print("STEP 2: Load/Create Paraphrased Questions")
    print(f"{'='*80}")
    
    paraphrased_samples = None
    
    # Check if we should use cached paraphrased questions
    if not FORCE_REPARAPHRASE:
        paraphrased_samples = load_paraphrased_questions()
        
        if paraphrased_samples and len(paraphrased_samples) == NUM_SAMPLES:
            print(f"✓ Using cached paraphrased questions ({len(paraphrased_samples)} samples)")
            print(f"  💡 To regenerate, set FORCE_REPARAPHRASE = True in the configuration")
        elif paraphrased_samples:
            print(f"⚠ Cached file has {len(paraphrased_samples)} samples, expected {NUM_SAMPLES}")
            print(f"  Will regenerate paraphrased questions...")
            paraphrased_samples = None
    else:
        print(f"🔄 FORCE_REPARAPHRASE is True - will regenerate paraphrased questions")
        if os.path.exists(PARAPHRASED_FILE):
            print(f"  (Existing cache file will be overwritten)")
    
    # Generate new paraphrased questions if needed
    if paraphrased_samples is None:
        print(f"\n🔨 Creating new paraphrased questions...")
        
        # Load original PIQA validation data
        print(f"\nLoading {DATASET_NAME} validation data...")
        original_samples = get_formatted_validation_data(DATASET_NAME, max_samples=NUM_SAMPLES)
        
        if not original_samples or len(original_samples) < NUM_SAMPLES:
            print(f"\n❌ ERROR: Could not load {NUM_SAMPLES} samples from {DATASET_NAME}")
            return
        
        print(f"✓ Loaded {len(original_samples)} original questions")
        
        # Paraphrase questions
        paraphrased_samples = await paraphrase_all_questions(original_samples)
        
        # Save paraphrased questions
        save_paraphrased_questions(paraphrased_samples)
    
    # Step 3: Evaluate all models on paraphrased questions
    print(f"\n{'='*80}")
    print("STEP 3: Evaluate Models on Paraphrased Questions")
    print(f"{'='*80}")
    
    total_questions = NUM_SAMPLES * len(MODELS)
    overall_pbar = tqdm(
        total=total_questions,
        desc="Overall Progress",
        unit="q",
        ncols=100
    )
    
    q5_results = {}
    
    for model in MODELS:
        results = await evaluate_model_on_paraphrased_questions(
            model,
            paraphrased_samples,
            progress_bar=overall_pbar
        )
        q5_results[model] = results
    
    overall_pbar.close()
    
    # Step 4: Compare Q5 with Q2
    print(f"\n{'='*80}")
    print("STEP 4: Compare Q5 with Q2")
    print(f"{'='*80}")
    
    comparisons = compare_q2_q5_results(q5_results, q2_results)
    
    # Print comparison tables
    print_comparison_tables(comparisons)
    
    # Calculate total time
    total_time = time.time() - overall_start_time
    end_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Start time:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Save results
    final_results = {
        'q5_results': q5_results,
        'comparisons': comparisons,
        'metadata': {
            'dataset': DATASET_NAME,
            'num_samples': NUM_SAMPLES,
            'models': MODELS,
            'paraphrase_model': PARAPHRASE_MODEL,
            'start_time': start_datetime.isoformat(),
            'end_time': end_datetime.isoformat(),
            'total_time_seconds': total_time
        }
    }
    
    try:
        with open(Q5_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {Q5_RESULTS_FILE}")
    except Exception as e:
        print(f"\n⚠ Error saving results: {e}")
    
    print(f"\n{'='*80}")
    print("Q5 EVALUATION COMPLETED!")
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
