# Structured Reasoning for Grammatical Feedback -A MAC-T5 Hybrid Approach-

This repository contains the code, data, and documentation for the MAC-T5 hybrid model, which integrates symbolic reasoning and neural generation to produce educational grammar feedback for second-language learners.

### Overview

Grammatical error correction (GEC) systems are highly accurate, yet most fail to explain why a sentence is incorrect. MAC-T5 addresses this by combining:

* T5: a transformer-based encoder-decoder for fluent generation.

* MAC (Memory, Attention, Control): a multi-step reasoning network that simulates instructor-like pedagogical thinking.

The model is trained on learner speech from the SLaTE 2025 Shared Task, and generates Chain-of-Thought (CoT) feedback: a 4-step explanation of the grammar rule, the mistake, the reason, and the correction.

### Contents


* Neutral Sentences ([Check the script](https://github.com/shuhashi0352/Japanese-Politeness-Classification/blob/main/Models/data_collection.ipynb)) (2000 total): Collected from Japanese Wikipedia, manually filtered to remove non-sentence lines like noun phrases and mathematical formulas.
