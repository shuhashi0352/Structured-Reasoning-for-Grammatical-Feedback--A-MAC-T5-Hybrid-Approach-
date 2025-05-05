# Structured Reasoning for Grammatical Feedback -A MAC-T5 Hybrid Approach-

This repository contains the code, data, and documentation for the MAC-T5 hybrid model, which integrates symbolic reasoning and neural generation to produce educational grammar feedback for second-language learners.

### Overview

Grammatical error correction (GEC) systems are highly accurate, yet most fail to explain why a sentence is incorrect. MAC-T5 addresses this by combining:

* T5: a transformer-based encoder-decoder for fluent generation.

* MAC (Memory, Attention, Control): a multi-step reasoning network that simulates instructor-like pedagogical thinking.

The model is trained on learner speech from the SLaTE 2025 Shared Task, and generates Chain-of-Thought (CoT) feedback: a 4-step explanation of the grammar rule, the mistake, the reason, and the correction.

### Model Architecture

**MAC-T5 Hybrid**

* Encoder: T5-small with prompt injection

* Reasoning: 4-step MAC-style attention loop

* Decoder: T5 for natural language feedback generation

**Reasoning Steps**

1. Identify the error

2. Attend to correction

3. Justify with grammar rules

4. Review explanation

### Evaluation

Two evaluation methods are used:

* BERTScore for semantic similarity with reference feedback

* LLM-based rubric evaluation for clarity, correctness, and educational value

Example Result (One-shot):
| Model  | BERTScore F1 | LLM Votes |
| ------ | ------------ | --------- |
| T5     | 0.8866       | 23        |
| MAC-T5 | 0.8168       | 3         |

MAC-T5 shows stronger gains with more data (10 votes in two-shot), suggesting good scalability.

### Dataset

* 11,655 error instances extracted from ASR transcripts

* Each instance includes:

    * Source sentence
  
    * Corrected sentence
  
    * Error tag
  
    * Error and correction phrases
  
    * Reference and CoT feedback

Training settings:

* One-shot: 50 instances (1 per error type)

* Two-shot: 100 instances (2 per error type)

* Dev/Test: Fixed 50 instances


See more details: ([Structured_Reasoning_for_Grammatical_Feedback__A_MAC_T5_Hybrid_Approach.pdf](https://github.com/shuhashi0352/Structured-Reasoning-for-Grammatical-Feedback--A-MAC-T5-Hybrid-Approach-/blob/main/Structured_Reasoning_for_Grammatical_Feedback__A_MAC_T5_Hybrid_Approach.pdf))
