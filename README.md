
# PokéEvaluator: Unleashing the Power of Model Fine-tuning on Pokémon Data

Dive deep into the realm of Pokémon with PokéEvaluator! This project seeks to compare the prowess of various models, both fine-tuned and vanilla, as they tackle the vibrant and intricate world of Pokémon data.
## Requirements

Ensure you have the required packages installed:

```
pip install -r requirements.txt
```

## Getting Started

### Notebooks

1. **Fine-tuning**: Use the `gpt_finetune.ipynb` notebook to generate training data and format it for fine-tuning and fine-tune away.
2. **Evaluation**: Use the `gpt_evaluate.ipynb` notebook to evaluate the models. This leverages the `PokemonGroundTruth` class to produce and score the data related to Pokémon.

### Model Interfacing

- **LLM**: Use the `llm.py` for the base language model structures and implementations.
- **OpenAI**: The `gpt_openai.py` for OpenAI-powered models within the project.
- **Llama2**: The `llama2_exllama.py` for Llama2-powered models within the project.

## Other Scripts

- `prompt.py`: Provides utilities for generating and managing prompts for the language models.
- `ground_truth_aiopoke.py`: Manages the generation and scoring of Pokémon data.
