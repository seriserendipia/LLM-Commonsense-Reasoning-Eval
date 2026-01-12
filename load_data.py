import os
import json
from datasets import Dataset, DatasetDict

# Import dataset path configuration from config.py
try:
    from config import DATASET_BASE_PATH
    BASE_PATH = DATASET_BASE_PATH
except ImportError:
    print("\n❌ Error: Could not find config.py file!")
    print("Please ensure config.py exists and DATASET_BASE_PATH is configured.")
    raise
except AttributeError:
    print("\n❌ Error: DATASET_BASE_PATH not found in config.py!")
    print("Please ensure DATASET_BASE_PATH is defined in your config.py.")
    raise

print("="*80)
print("LOADING COMMONSENSE REASONING BENCHMARKS")
print(f"Dataset Path: {BASE_PATH}")
print("="*80)

# ============================================================================
# Part 1: Load Arrow-based datasets (5 datasets)
# ============================================================================

# List of Arrow-based datasets
arrow_datasets = {
    "CommonsenseQA": "commonsenseqa",
    "HellaSwag": "hellaswag",
    "PIQA": "piqa",
    "SocialIQA": "socialiqa",
    # "WinoGrande": "winogrande"
}

# Dictionary to store loaded datasets
loaded_datasets = {}

for dataset_name, folder_name in arrow_datasets.items():
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name}...")
    print(f"{'='*80}")
    
    dataset_path = os.path.join(BASE_PATH, folder_name)
    
    try:
        # Load each split separately since they are in separate folders
        # Check which splits exist for this dataset
        available_splits = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d))]
        
        dataset_splits = {}
        
        for split in available_splits:
            split_path = os.path.join(dataset_path, split)
            arrow_file = os.path.join(split_path, "data.arrow")
            
            if os.path.exists(arrow_file):
                # Load the Arrow file as a Dataset
                dataset_splits[split] = Dataset.from_file(arrow_file)
                print(f"  ✓ Loaded {split} split from {arrow_file}")
        
        # Create a DatasetDict from the loaded splits
        if dataset_splits:
            dataset = DatasetDict(dataset_splits)
            loaded_datasets[dataset_name] = dataset
            
            # Print dataset summary
            print(f"\n{dataset_name} Dataset Summary:")
            print(dataset)
            
            # # Print detailed statistics for each split
            # print(f"\nDetailed Statistics for {dataset_name}:")
            # for split_name, split_data in dataset.items():
            #     print(f"  - {split_name}: {len(split_data)} rows")
            #     print(f"    Features: {list(split_data.features.keys())}")
        else:
            print(f"  ⚠ No valid splits found for {dataset_name}")
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Part 2: Load TG-CSR (JSON-based dataset)
# ============================================================================

print(f"\n{'='*80}")
print("Loading TG-CSR (JSON-based dataset)...")
print(f"{'='*80}")

tg_csr_path = os.path.join(BASE_PATH, "tgcsr")

# Dictionary to store TG-CSR data
tg_csr_data = {}

# Load both train and validation splits
for split in ["train", "validation"]:
    split_path = os.path.join(tg_csr_path, split)
    
    try:
        print(f"\n{'='*60}")
        print(f"Loading TG-CSR {split} split...")
        print(f"{'='*60}")
        
        # Load all JSON files for this split
        questions_file = os.path.join(split_path, "questions.json")
        qa_pairs_file = os.path.join(split_path, "QA-pair.json")
        candidate_answers_file = os.path.join(split_path, "candidate-answers.json")
        labels_file = os.path.join(split_path, "labels.lst")
        questions_tags_file = os.path.join(split_path, "questions-tags.json")
        answers_tags_file = os.path.join(split_path, "candidate-answers-tags.json")
        
        # Load questions.json (contains context, theme, and nested tasks)
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Load QA-pair.json (list of question-answer pairings)
        with open(qa_pairs_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        # Load candidate-answers.json (list of all possible answers)
        with open(candidate_answers_file, 'r', encoding='utf-8') as f:
            candidate_answers = json.load(f)
        
        # Load labels.lst (correct answer indices, one per line)
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        
        # Load questions-tags.json (semantic tags for questions)
        with open(questions_tags_file, 'r', encoding='utf-8') as f:
            questions_tags = json.load(f)
        
        # Load candidate-answers-tags.json (semantic tags for answers)
        with open(answers_tags_file, 'r', encoding='utf-8') as f:
            answers_tags = json.load(f)
        
        # Store all data for this split
        tg_csr_data[split] = {
            'questions_data': questions_data,
            'qa_pairs': qa_pairs,
            'candidate_answers': candidate_answers,
            'labels': labels,
            'questions_tags': questions_tags,
            'answers_tags': answers_tags
        }
        
        # Print detailed statistics
        print(f"\n  ✓ Loaded {split} split successfully")
        print(f"\n  📊 Split Statistics:")
        print(f"    - Total QA pairs: {len(qa_pairs)}")
        print(f"    - Total labels: {len(labels)}")
        print(f"    - Total candidate answers: {len(candidate_answers)}")
        print(f"    - Unique questions (by ID): {len(set(pair['id'] for pair in qa_pairs))}")
        print(f"    - Question tags: {len(questions_tags)}")
        print(f"    - Answer tags: {len(answers_tags)}")
        
        # Show the structure of questions.json
        print(f"\n  📋 Questions.json Structure:")
        print(f"    - Keys: {list(questions_data.keys())}")
        print(f"    - Context: {questions_data.get('context', 'N/A')}")
        print(f"    - Theme preview: {questions_data.get('theme', 'N/A')[:100]}...")
        if 'tasks' in questions_data:
            print(f"    - Number of task groups: {len(questions_data['tasks'])}")
            if len(questions_data['tasks']) > 0:
                print(f"    - Questions in first task group: {len(questions_data['tasks'][0])}")
        
        # Show sample QA pair structure
        if len(qa_pairs) > 0:
            print(f"\n  🔍 Sample QA-pair (first entry):")
            print(f"    {qa_pairs[0]}")
        
        # Show sample candidate answer
        if len(candidate_answers) > 0:
            print(f"\n  🔍 Sample Candidate Answer (first entry):")
            print(f"    {candidate_answers[0]}")
        
        # Show distribution of tags
        if len(questions_tags) > 0:
            tag_distribution = {}
            for item in questions_tags:
                tag = item.get('tag', 'Unknown')
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
            print(f"\n  🏷️  Question Tag Distribution:")
            for tag, count in sorted(tag_distribution.items()):
                print(f"    - {tag}: {count} questions")
        
    except Exception as e:
        print(f"\n  ❌ Error loading TG-CSR {split} split: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Part 3: Summary Report
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY REPORT")
print(f"{'='*80}")

print("\n1. Arrow-based Datasets:")
for dataset_name in arrow_datasets.keys():
    if dataset_name in loaded_datasets:
        dataset = loaded_datasets[dataset_name]
        print(f"\n{dataset_name}:")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} examples")

print("\n2. TG-CSR (JSON-based):")
for split, data in tg_csr_data.items():
    num_qa_pairs = len(data['qa_pairs'])
    num_unique_questions = len(set(pair['id'] for pair in data['qa_pairs']))
    print(f"  - {split}: {num_qa_pairs} QA pairs ({num_unique_questions} unique questions)")

print(f"\n{'='*80}")
print("Dataset loading completed!")
print(f"{'='*80}")

# ============================================================================
# Helper Functions for Arrow-based Datasets
# ============================================================================

def get_dataset_samples(dataset, split='train', n=0):
    """
    Get the first n samples from an Arrow-based dataset.
    
    Args:
        dataset (DatasetDict): The loaded dataset (e.g., CommonsenseQA)
        split (str): The split to use ('train', 'validation', 'test')
        n (int): Number of samples to return. If n=0 or n>=total, return all samples
    
    Returns:
        list: List of sample dictionaries
    
    Example:
        >>> samples = get_dataset_samples(loaded_datasets['CommonsenseQA'], split='train', n=5)
        >>> print(samples[0])  # First sample
    """
    # Check if split exists
    if split not in dataset:
        available_splits = list(dataset.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
    
    split_data = dataset[split]
    total_samples = len(split_data)
    
    # If n=0 or n >= total samples, return all
    if n <= 0 or n >= total_samples:
        n = total_samples
    
    # Get first n samples
    samples = []
    for i in range(n):
        samples.append(split_data[i])
    
    return samples


def print_dataset_sample(sample, dataset_name="Dataset", index=0):
    """
    Pretty print a single sample from an Arrow-based dataset.
    
    Args:
        sample (dict): A single sample dictionary
        dataset_name (str): Name of the dataset for display
        index (int): Index of this sample
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name} - Sample #{index}")
    print(f"{'='*80}")
    for key, value in sample.items():
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"{key}: {value_str}")
    print(f"{'='*80}")


def get_all_datasets_samples(loaded_datasets, split='train', n=5):
    """
    Get samples from all loaded Arrow-based datasets.
    
    Args:
        loaded_datasets (dict): Dictionary of loaded datasets
        split (str): The split to use ('train', 'validation', 'test')
        n (int): Number of samples per dataset. If n=0, return all samples
    
    Returns:
        dict: Dictionary mapping dataset_name -> list of samples
    
    Example:
        >>> all_samples = get_all_datasets_samples(loaded_datasets, split='train', n=3)
        >>> print(all_samples['CommonsenseQA'][0])
    """
    all_samples = {}
    
    for dataset_name, dataset in loaded_datasets.items():
        # Check if split exists for this dataset
        if split in dataset:
            try:
                samples = get_dataset_samples(dataset, split=split, n=n)
                all_samples[dataset_name] = samples
            except Exception as e:
                print(f"Error getting samples from {dataset_name}: {e}")
        else:
            print(f"Warning: {dataset_name} does not have '{split}' split. Skipping.")
    
    return all_samples

# Get first 3 samples from arrow-based datasets
all_arrow_samples = get_all_datasets_samples(loaded_datasets, split='train', n=3)
for dataset_name, samples in all_arrow_samples.items():
    for i, sample in enumerate(samples):
        print_dataset_sample(sample, dataset_name=dataset_name, index=i)    

# ============================================================================
# Helper Functions for TG-CSR Data Access
# ============================================================================

def get_question_text_by_id(questions_data, question_id):
    """
    Find the question text by question ID from the nested tasks structure.
    
    Args:
        questions_data (dict): The questions.json data containing 'tasks'
        question_id (int): The ID of the question to find
    
    Returns:
        str or None: The question text if found, None otherwise
    """
    tasks = questions_data.get('tasks', [])
    if tasks and len(tasks) > 0:
        for task_group in tasks:
            for q in task_group:
                if q.get('id') == question_id:
                    return q.get('question')
    return None


def get_answer_text_by_index(candidate_answers, answer_index):
    """
    Find the answer text by answer index.
    
    Args:
        candidate_answers (list): List of candidate answer dictionaries
        answer_index (int): The index of the answer to find
    
    Returns:
        str or None: The answer text if found, None otherwise
    """
    for ans in candidate_answers:
        if ans.get('index') == answer_index:
            return ans.get('answer')
    return None


def get_tag_by_id(tags_list, item_id, id_key='id'):
    """
    Find the tag for a given item ID.
    
    Args:
        tags_list (list): List of tag dictionaries
        item_id (int): The ID to search for
        id_key (str): The key name for the ID field (default: 'id')
    
    Returns:
        str or None: The tag if found, None otherwise
    """
    for tag_item in tags_list:
        if tag_item.get(id_key) == item_id:
            return tag_item.get('tag')
    return None


def get_tgcsr_example(split_data, qa_index=0):
    """
    Get a single complete example from TG-CSR dataset by QA pair index.
    
    Args:
        split_data (dict): The TG-CSR split data containing all JSON files
        qa_index (int): The index of the QA pair to retrieve (default: 0)
    
    Returns:
        dict: A dictionary containing all information about this QA pair, or None if invalid
    """
    # Validate input
    if not split_data or qa_index >= len(split_data.get('qa_pairs', [])):
        return None
    
    # Get QA pair
    qa_pair = split_data['qa_pairs'][qa_index]
    question_id = qa_pair['id']
    answer_index = qa_pair['index']
    
    # Get label
    label = split_data['labels'][qa_index]
    
    # Get question text
    question_text = get_question_text_by_id(
        split_data['questions_data'], 
        question_id
    )
    
    # Get answer text
    answer_text = get_answer_text_by_index(
        split_data['candidate_answers'], 
        answer_index
    )
    
    # Get question tag
    question_tag = get_tag_by_id(
        split_data['questions_tags'], 
        question_id, 
        id_key='id'
    )
    
    # Get answer tag
    answer_tag = get_tag_by_id(
        split_data['answers_tags'], 
        answer_index, 
        id_key='index'
    )
    
    # Get context
    context = split_data['questions_data'].get('context', '')
    theme = split_data['questions_data'].get('theme', '')
    
    return {
        'context': context,
        'theme': theme,
        'question_id': question_id,
        'question': question_text,
        'question_tag': question_tag,
        'answer_index': answer_index,
        'answer': answer_text,
        'answer_tag': answer_tag,
        'label': label,
        'is_correct': label == 1
    }


def get_tgcsr_samples(tg_csr_data, split='train', n=0):
    """
    Get the first n samples from TG-CSR dataset.
    
    Args:
        tg_csr_data (dict): The loaded TG-CSR data containing 'train' and 'validation' splits
        split (str): The split to use ('train', 'validation')
        n (int): Number of samples to return. If n=0 or n>=total, return all samples
    
    Returns:
        list: List of sample dictionaries, each containing question, answer, label, etc.
    
    Example:
        >>> samples = get_tgcsr_samples(tg_csr_data, split='train', n=5)
        >>> print(samples[0])  # First QA pair
    """
    # Check if split exists
    if split not in tg_csr_data:
        available_splits = list(tg_csr_data.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
    
    split_data = tg_csr_data[split]
    total_qa_pairs = len(split_data.get('qa_pairs', []))
    
    # If n=0 or n >= total samples, return all
    if n <= 0 or n >= total_qa_pairs:
        n = total_qa_pairs
    
    # Get first n samples
    samples = []
    for i in range(n):
        example = get_tgcsr_example(split_data, qa_index=i)
        if example:
            samples.append(example)
    
    return samples




def print_tgcsr_example(example, title="TG-CSR Example"):
    """
    Pretty print a TG-CSR example.
    
    Args:
        example (dict): Example dictionary returned by get_tgcsr_example()
        title (str): Title to display
    """
    if example is None:
        print("Invalid example")
        return
    
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"\nContext: {example['context']}")
    print(f"\nQuestion ID: {example['question_id']}")
    print(f"Question: {example['question']}")
    print(f"Question Tag: {example['question_tag']}")
    print(f"\nCandidate Answer Index: {example['answer_index']}")
    print(f"Candidate Answer: {example['answer']}")
    print(f"Answer Tag: {example['answer_tag']}")
    print(f"\nLabel: {example['label']} (0=correct, 1=incorrect)")
    print(f"Is this answer correct? {'✅ YES' if example['is_correct'] else '❌ NO'}")
    print(f"{'='*80}")


# ============================================================================
# Optional: Show complete examples from TG-CSR to understand the data structure
# ============================================================================

if 'train' in tg_csr_data and len(tg_csr_data['train']['qa_pairs']) > 0:
    # Example 1: Get first 3 samples using the new function
    print(f"\n{'='*80}")
    print("TG-CSR Examples (First 3 QA pairs from train split)")
    print(f"{'='*80}")
    
    samples = get_tgcsr_samples(tg_csr_data, split='train', n=3)
    for i, sample in enumerate(samples):
        print(f"\nSample {i}:")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Correct: {'✅ YES' if sample['is_correct'] else '❌ NO'}")


# ============================================================================
# Part 4: Format Unification Functions (for Q2 evaluation)
# ============================================================================

def format_commonsenseqa_sample(sample):
    """
    Convert CommonsenseQA sample to unified format.
    
    Original format:
        - question: str (the question text)
        - choices: dict with 'label' and 'text' lists
        - answerKey: str (e.g., 'A', 'B', 'C', 'D', 'E')
    
    Unified format:
        - question: str
        - choices: list of str (choice texts)
        - correct_answer: int (index of correct answer, 0-based)
        - is_multi_choice: bool (False for single-choice)
    
    Args:
        sample (dict): Original CommonsenseQA sample
    
    Returns:
        dict: Unified format sample
    """
    return {
        'question': sample['question'],
        'choices': sample['choices']['text'],
        'correct_answer': sample['choices']['label'].index(sample['answerKey']),
        'is_multi_choice': False  # Single-choice question
    }


def format_hellaswag_sample(sample):
    """
    Convert HellaSwag sample to unified format.
    
    Original format:
        - ctx: str (context)
        - endings: list of str (4 possible endings)
        - label: str (index as string, e.g., '0', '1', '2', '3')
    
    Unified format:
        - question: str (context serves as question)
        - choices: list of str (the endings)
        - correct_answer: int (index of correct answer, 0-based)
        - is_multi_choice: bool (False for single-choice)
    
    Args:
        sample (dict): Original HellaSwag sample
    
    Returns:
        dict: Unified format sample
    """
    return {
        'question': sample['ctx'],
        'choices': sample['endings'],
        'correct_answer': int(sample['label']),
        'is_multi_choice': False  # Single-choice question
    }


def format_piqa_sample(sample):
    """
    Convert PIQA sample to unified format.
    
    Original format:
        - goal: str (the question/goal)
        - sol1: str (solution 1)
        - sol2: str (solution 2)
        - label: int (0 or 1, indicating correct solution)
    
    Unified format:
        - question: str
        - choices: list of str (two solutions)
        - correct_answer: int (index of correct answer, 0-based)
        - is_multi_choice: bool (False for single-choice)
    
    Args:
        sample (dict): Original PIQA sample
    
    Returns:
        dict: Unified format sample
    """
    return {
        'question': sample['goal'],
        'choices': [sample['sol1'], sample['sol2']],
        'correct_answer': int(sample['label']),
        'is_multi_choice': False  # Single-choice question
    }


def format_socialiqa_sample(sample):
    """
    Convert SocialIQA sample to unified format.
    
    Original format:
        - context: str (context)
        - question: str (the question)
        - answerA, answerB, answerC: str (three answer choices)
        - label: str ('1', '2', or '3')
    
    Unified format:
        - question: str (context + question)
        - choices: list of str (three answers)
        - correct_answer: int (index of correct answer, 0-based)
    
    Args:
        sample (dict): Original SocialIQA sample
    
    Returns:
        dict: Unified format sample
    """
    # Combine context and question for full question text
    full_question = f"{sample['context']} {sample['question']}"
    
    return {
        'question': full_question,
        'choices': [sample['answerA'], sample['answerB'], sample['answerC']],
        'correct_answer': int(sample['label']) - 1,  # Convert from 1-based to 0-based
        'is_multi_choice': False  # Single-choice question
    }


def format_tgcsr_sample(sample):
    """
    Convert TG-CSR sample to unified format.
    
    Note: TG-CSR is already partially formatted by get_tgcsr_example().
    This function handles the conversion for the evaluation format.
    
    For TG-CSR, we need to group all candidate answers for the same question.
    This is different from other datasets where each sample is independent.
    
    Args:
        sample (dict): TG-CSR sample from get_tgcsr_example()
    
    Returns:
        dict: Unified format sample (partial - needs grouping)
    """
    # Note: This is a simplified version. 
    # TG-CSR requires special handling in get_formatted_validation_data()
    # because multiple QA pairs share the same question.
    return {
        'question': sample['question'],
        'choices': [sample['answer']],  # Single choice (will be grouped later)
        'correct_answer': 0 if sample['is_correct'] else -1,
        'question_id': sample['question_id'],
        'answer_index': sample['answer_index']
    }


def get_formatted_validation_data(dataset_name, max_samples=0, print_examples=False):
    """
    Get validation data for a specific dataset in unified format.
    
    This function:
    1. Loads the validation split of the specified dataset
    2. Converts samples to unified format
    3. Returns a list of formatted samples
    
    Unified format for each sample:
        - question: str (the question text)
        - choices: list of str (answer choices)
        - correct_answer: int (index of correct answer, 0-based)
    
    Args:
        dataset_name (str): Name of the dataset 
            ('CommonsenseQA', 'HellaSwag', 'PIQA', 'SocialIQA', 'TG-CSR')
        max_samples (int): Maximum number of samples to return.
            If 0, return all samples. Default: 0
    
    Returns:
        list of dict: List of formatted samples
    
    Example:
        >>> data = get_formatted_validation_data('CommonsenseQA', max_samples=3)
        >>> print(data[0])
        {
            'question': 'What is ...',
            'choices': ['option1', 'option2', ...],
            'correct_answer': 2
        }
    """
    formatted_samples = []
    
    # Handle Arrow-based datasets
    if dataset_name in loaded_datasets:
        dataset = loaded_datasets[dataset_name]
        
        # Check if validation split exists
        if 'validation' not in dataset:
            raise ValueError(f"{dataset_name} does not have a validation split")
        
        validation_data = dataset['validation']
        total_samples = len(validation_data)
        
        # Determine number of samples to process
        num_samples = total_samples if max_samples <= 0 else min(max_samples, total_samples)
        
        print(f"\n{'='*80}")
        print(f"Loading {dataset_name} validation data (formatted)")
        print(f"{'='*80}")
        print(f"  Total samples: {total_samples}")
        print(f"  Samples to process: {num_samples}")
        
        # Select appropriate formatting function
        format_func = None
        if dataset_name == 'CommonsenseQA':
            format_func = format_commonsenseqa_sample
        elif dataset_name == 'HellaSwag':
            format_func = format_hellaswag_sample
        elif dataset_name == 'PIQA':
            format_func = format_piqa_sample
        elif dataset_name == 'SocialIQA':
            format_func = format_socialiqa_sample
        
        # Format samples
        for i in range(num_samples):
            sample = validation_data[i]
            formatted_sample = format_func(sample)
            formatted_samples.append(formatted_sample)
        
        print(f"  ✓ Loaded and formatted {len(formatted_samples)} samples")
        
        # Print first sample as example
        if len(formatted_samples) > 0:
            print(f"\n  📝 Example (first sample):")
            print(f"    Question: {formatted_samples[0]['question'][:100]}...")
            print(f"    Choices: {len(formatted_samples[0]['choices'])} options")
            print(f"    Correct answer index: {formatted_samples[0]['correct_answer']}")
    
    # Handle TG-CSR (JSON-based dataset)
    elif dataset_name == 'TG-CSR':
        if 'validation' not in tg_csr_data:
            raise ValueError("TG-CSR validation split not loaded")
        
        split_data = tg_csr_data['validation']
        
        print(f"\n{'='*80}")
        print(f"Loading TG-CSR validation data (formatted)")
        print(f"{'='*80}")
        
        # TG-CSR has a special structure: multiple QA pairs per question
        # We need to group candidate answers by question_id
        
        # First, get all QA pairs
        all_qa_pairs = split_data['qa_pairs']
        total_qa_pairs = len(all_qa_pairs)
        
        # Group by question_id
        question_groups = {}
        for qa_pair_seq, qa_pair in enumerate(all_qa_pairs):
            question_id = qa_pair['id']
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(qa_pair_seq)

        print(f"  Total QA pairs: {total_qa_pairs}")
        print(f"  Unique questions: {len(question_groups)}")
        
        # Process each unique question
        question_ids = list(question_groups.keys())
        num_questions = len(question_ids) if max_samples <= 0 else min(max_samples, len(question_ids))
        
        print(f"  Questions to process: {num_questions}")
        
        for q_idx in range(num_questions):
            question_id = question_ids[q_idx]
            qa_pair_seqs = question_groups[question_id]
            
            # Get all candidate answers for this question
            choices = []
            correct_idx = []
            question_text = None

            for i, qa_idx in enumerate(qa_pair_seqs):
                example = get_tgcsr_example(split_data, qa_index=qa_idx)
                if example:
                    if question_text is None:
                        # Combine context + theme + question for complete question text
                        context = example.get('context', '').strip()
                        theme = example.get('theme', '').strip()
                        question = example.get('question', '').strip()
                        
                        # Build complete question with context
                        parts = []
                        if context:
                            parts.append(f"Context: {context}")
                        if theme:
                            parts.append(f"Theme: {theme}")
                        if question:
                            parts.append(f"Question: {question}")
                        
                        question_text = '\n'.join(parts)
                    
                    choices.append(example['answer'])
                    if example['is_correct']:
                        correct_idx.append(i)
            
            # Create formatted sample
            if question_text and choices:
                # Store all correct answer indices (can be multiple or empty)
                formatted_sample = {
                    'question': question_text,
                    'choices': choices,
                    'correct_answer': correct_idx,  # List of all correct indices (can be empty)
                    'question_id': question_id,
                    'is_multi_choice': True  # TG-CSR is multi-choice question
                }
                formatted_samples.append(formatted_sample)
        
        print(f"  ✓ Loaded and formatted {len(formatted_samples)} questions")
        
        if print_examples:
            for i, sample in enumerate(formatted_samples):
                print(f"\n  📝 Example (question {i+1}):")
                print(f"    Question: {sample['question']}")
                print(f"    Choices: {len(sample['choices'])} options")
                for idx, choice in enumerate(sample['choices']):
                    print(f"    Choice {idx}: {choice}")
                print(f"    Correct answer index: {sample['correct_answer']}")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return formatted_samples


# ============================================================================
# Testing the format unification (optional demo)
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("TESTING FORMAT UNIFICATION")
    print(f"{'='*80}")
    
    # Test with 2 samples from each dataset
    test_datasets = [
        'CommonsenseQA',
        'HellaSwag',
        'PIQA',
        'SocialIQA',
        'TG-CSR'
    ]

    for dataset_name in test_datasets:
        try:
            formatted_data = get_formatted_validation_data(dataset_name, max_samples=5, print_examples=True)
            print(f"\n✅ {dataset_name}: Successfully formatted {len(formatted_data)} samples")
        except Exception as e:
            print(f"\n❌ {dataset_name}: Error - {e}")
    
    print(f"\n{'='*80}")
    print("Format unification testing completed!")
    print(f"{'='*80}")

