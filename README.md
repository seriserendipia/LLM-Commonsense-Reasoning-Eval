# LLM Commonsense Reasoning Evaluation

This project evaluates Large Language Models (LLMs) on multiple commonsense reasoning benchmarks.

---

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install openai datasets tenacity tqdm
```

### 2. Configure Settings

**⚠️ Important: All configurations are done in the `config.py` file only.**

Open the `config.py` file and configure the following two items:

#### (1) API Key Configuration

Get your API key from [OpenRouter](https://openrouter.ai/settings/keys) and set it in `config.py`:

```python
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### (2) Dataset Path Configuration

Set the dataset path in `config.py`:

**Option A (Recommended):** Place the `nlp-reasoning-benchmarks` folder in the project directory.
```python
# Default configuration, no changes needed if datasets are in the project folder.
DATASET_BASE_PATH = os.path.join(os.path.dirname(__file__), "nlp-reasoning-benchmarks")
```

**Option B:** If the datasets are in another location, modify the path to an absolute path:
```python
DATASET_BASE_PATH = r"D:\Your\Path\To\nlp-reasoning-benchmarks"
```

### 3. Verify Configuration

Run the following command to test if the configuration is correct:

```bash
python config.py
```

If you see "✅ Configuration validated successfully!", the configuration is correct.

---

## 📁 Project Structure

```
HW2/
├── config.py                  # 🔧 Configuration file (API Key + Dataset Path)
├── api_key_loader.py          # API key loading utility
├── load_data.py               # Dataset loading functions
├── Q2_async.py                # Main evaluation (async)
├── Q3.py                      # Robustness test (option removal)
├── Q4_async.py                # Evaluation with "I don't know" option
├── Q5.py                      # Question paraphrasing test
├── nlp-reasoning-benchmarks/  # Dataset folder (to be prepared by the user)
│   ├── commonsense_qa/
│   ├── hellaswag/
│   ├── piqa/
│   ├── social_i_qa/
│   └── tgcsr/
└── result data/               # Evaluation results (auto-generated)
```

---

## 📊 Dataset Structure

Ensure your `nlp-reasoning-benchmarks` folder contains the following datasets:

```
nlp-reasoning-benchmarks/
├── commonsense_qa/
│   ├── train/
│   │   └── data.arrow
│   └── validation/
│       └── data.arrow
├── hellaswag/
│   ├── train/
│   │   └── data.arrow
│   └── validation/
│       └── data.arrow
├── piqa/
│   ├── train/
│   │   └── data.arrow
│   └── validation/
│       └── data.arrow
├── social_i_qa/
│   ├── train/
│   │   └── data.arrow
│   └── validation/
│       └── data.arrow
└── tgcsr/
    ├── train/
    │   ├── questions.json
    │   ├── QA-pair.json
    │   ├── candidate-answers.json
    │   ├── labels.lst
    │   ├── questions-tags.json
    │   └── candidate-answers-tags.json
    └── validation/
        └── (same files as train)
```

---

## 🚀 Usage

### Q2: Main Evaluation (Async)

Evaluate 3 LLMs on 5 datasets:

```bash
python Q2_async.py
```

**Models:**
- Gemini 1.5 Flash
- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct

**Datasets:**
- CommonsenseQA
- HellaSwag
- PIQA
- SocialIQA
- TGCSR

### Q3: Robustness Test

Test model performance after removing a correct option:

```bash
python Q3.py
```

### Q4: "I don't know" Option Evaluation

Evaluate with an added "I don't know" option:

```bash
python Q4_async.py
```

### Q5: Question Paraphrasing Test

Test the impact of question paraphrasing on model performance:

```bash
python Q5.py
```

---

## 📈 Results

All evaluation results are saved in the `result data/` folder:

- `Q2_async_final_results.json` - Main evaluation results
- `Q3_final_results.json` - Robustness test results
- `Q4_async_final_results.json` - "I don't know" option evaluation results
- `Q5_final_results.json` - Question paraphrasing test results

---

