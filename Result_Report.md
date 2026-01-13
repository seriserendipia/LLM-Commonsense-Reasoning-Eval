This report consolidates five research stages evaluating the performance and robustness of Large Language Models (LLMs) on commonsense reasoning benchmarks.

---

# Consolidated LLM Commonsense Reasoning Evaluation Report

## 1. Paraphrase Robustness on PIQA: Q5 vs Q2

This section analyzes how paraphrasing affects model performance on the Physical Interaction QA (PIQA) dataset.

### 1.1 Overview

* 
**Dataset**: PIQA (validation split), 50 sampled questions.


* 
**Paraphraser**: `qwen/qwen-2.5-72b-instruct`.


* 
**Models**: `openai/gpt-oss-120b`, `google/gemini-2.0-flash-lite-001`, `qwen/qwen-2.5-72b-instruct`.



### 1.2 Accuracy and Consistency Analysis

The following tables compare the original (Q2) and paraphrased (Q5) performance.

**Accuracy Comparison**
| Model | Q2 Acc % | Q5 Acc % | Diff % |
| :--- | :--- | :--- | :--- |
| openai/gpt-oss-120b | 8.00 | 8.00 | +0.00 |
| google/gemini-2.0-flash-lite-001 | 92.00 | 90.00 | -2.00 |
| qwen/qwen-2.5-72b-instruct | 94.00 | 92.00 | -2.00 |

**Consistency (Same Answer Rate)**
| Model | Same Answer | Rate % |
| :--- | :--- | :--- |
| openai/gpt-oss-120b | 44/50 | 88.00 |
| google/gemini-2.0-flash-lite-001 | 43/50 | 86.00 |
| qwen/qwen-2.5-72b-instruct | 47/50 | 94.00 |

**Correctness Transitions**
| Model | Both Correct | Q2✓→Q5✗ | Q2✗→Q5✓ | Both Wrong |
| :--- | :--- | :--- | :--- | :--- |
| openai/gpt-oss-120b | 1 | 3 | 3 | 43 |
| google/gemini-2.0-flash-lite-001 | 42 | 4 | 3 | 1 |
| qwen/qwen-2.5-72b-instruct | 46 | 1 | 0 | 3 |

### 1.3 Findings and Interpretation

* 
**Accuracy Shift**: Strong models saw minor declines (-2 pp), while OpenAI remained at a low baseline.


* 
**Robustness**: Qwen showed the highest consistency (94%), indicating strong stability against wording changes.


* 
**Error Drivers**: Surface wording changes can flip model decisions even when meaning is preserved. For example, the trade-off in "perfectly golden pancakes" or ambiguity around "fire" and "melt" triggered mistakes.



---

## 2. General Performance Evaluation on Validation Sets

This section summarizes the accuracy and efficiency of three LLMs across five standard benchmarks.

### 2.1 Executive Summary

* 
**Top Performer**: `qwen/qwen-2.5-72b-instruct` achieved the highest accuracy (80.96% - 94.34%).


* 
**Strong Performance**: `google/gemini-2.0-flash-lite-001` was the most time-efficient and performed well (77.94% - 91.40%).


* 
**Evaluation Anomaly**: `openai/gpt-oss-120b` scores were artificially deflated (6.29% - 10.16%) due to verbose responses exceeding the `max_tokens=20` limit.



### 2.2 Accuracy Analysis (%)

| Model | CommonsenseQA | HellaSwag | PIQA | SocialIQA | TG-CSR |
| --- | --- | --- | --- | --- | --- |
| openai/gpt-oss-120b | 10.16% | 9.61% | 8.32% | 6.29% | 0.00% |
| google/gemini-2.0-flash-lite-001 | 82.96% | 84.76% | 91.40% | 77.94% | 25.00% |
| qwen/qwen-2.5-72b-instruct | 87.14% | 88.64% | 94.34% | 80.96% | 25.00% |



### 2.3 Failure Analysis: Length-Limited Rate (%)

The high rate of truncated responses explains the OpenAI model's low score.
| Model | CommonsenseQA | HellaSwag | PIQA | SocialIQA | TG-CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| openai/gpt-oss-120b | 65.11% | 67.84% | 61.37% | 69.60% | 75.00% |
| google/gemini-2.0-flash-lite-001 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen/qwen-2.5-72b-instruct | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |



---

## 3. Robustness on CommonsenseQA with 4-Choice Options

This experiment analyzed if reducing distractor options (incorrect choices) improves reasoning performance.

### 3.1 Comparative Performance

Accuracy improved for models unaffected by technical parsing issues.
| Model | Acc (5 Options) | Acc (4 Options) | Improvement (pp) |
| :--- | :--- | :--- | :--- |
| openai/gpt-oss-120b | 14.09% | 14.09% | 0.00 |
| google/gemini-2.0-flash-lite-001 | 82.64% | 85.01% | +2.37 |
| qwen/qwen-2.5-72b-instruct | 87.06% | 89.03% | +1.97 |



### 3.2 Key Findings

* 
**Simplified Reasoning**: Reducing the number of choices makes tasks easier by removing confusing distractors.


* 
**Cognitive Load**: Even for high-capability models, additional incorrect options increase the probability of error.



---

## 4. Robustness with an "I don't know" (IDK) Option

This section assesses model behavior when given an explicit way to express uncertainty.

### 4.1 Accuracy Impact (Q2 vs Q4)

The introduction of an IDK option generally led to significant accuracy drops as models chose to abstain rather than guess correctly.

| Model | Dataset | Acc (Q2 Forced) | Acc (Q4 IDK) | Change |
| --- | --- | --- | --- | --- |
| google/gemini-2.0-flash | SocialIQA | 77.94% | 71.55% | -6.39% |
| google/gemini-2.0-flash | TG-CSR | 25.00% | 0.00% | -25.00% |
| qwen/qwen-2.5-72b | PIQA | 94.34% | 88.08% | -6.26% |
| qwen/qwen-2.5-72b | TG-CSR | 25.00% | 0.00% | -25.00% |



### 4.2 Behavioral Personalities

Models exhibited distinct behaviors when expressing uncertainty:

* 
**The Cautious (Gemini)**: Frequently used IDK, prioritizing honesty over potential accuracy.


* 
**The Confident (Qwen)**: Moderately cautious but abstained less often than Gemini.


* 
**The Stubborn (OpenAI)**: Almost never used the IDK option despite low performance.



### 4.3 The Uncertainty Trade-Off

The accuracy drop is driven by "Lost Confidence," where models flag questions they actually know how to answer correctly.
| Model (PIQA) | Improved Honesty (Wrong → IDK) | Lost Confidence (Right → IDK) |
| :--- | :--- | :--- |
| google/gemini-2.0-flash | 31 | 107 |
| qwen/qwen-2.5-72b | 19 | 117 |



---

## 5. Dataset Overview and Samples

This final section provides a profile of the six benchmarks used in the evaluation.

### 5.1 Dataset Profiles

* 
**CommonsenseQA**: Multiple-choice QA requiring broad commonsense (1,221 validation examples).


* 
**HellaSwag**: Natural Language Inference focusing on continuing context (10,042 validation examples).


* 
**PIQA**: Focuses on physical interaction with everyday objects (1,838 validation examples).


* 
**SocialIQA**: Probes emotional and social intelligence (1,954 validation examples).


* 
**TG-CSR**: Task-grounded reasoning in practical scenarios (77 validation pairs).



### 5.2 Representative Samples

* **CommonsenseQA**: "Sammy wanted to go to where the people were. Where might he go?" (Choices: race track, populated areas, the desert, etc.) .


* 
**PIQA**: "To permanently attach metal legs to a chair, you can..." (Solutions: Weld the metal or nail the metal).


* **SocialIQA**: "Cameron decided to have a barbecue and gathered her friends together. How would Others feel?" (Answers: like attending, like staying home, etc.) .



