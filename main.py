import asyncio
import json

import pandas as pd
from dotenv import load_dotenv

from evaluator import PokemonEvaluator
from fine_tuner import FineTuner
from ground_truth_aiopoke import PokemonGroundTruth
from llama2_exllama import Llama2ExLlama
from gpt4_openai import GPT4OpenAI
from prompt import dataframe_to_table_string


if __name__ == "__main__":
    load_dotenv()

    # with open("results.csv") as results_file:
    #     results = pd.read_csv(results_file)
    #     print(results)

    # all_pokemon_list contains all 947 Pokémon names
    with open("pokemon_names.json") as pokemons_file:
        all_pokemon_list = json.load(pokemons_file)
    # max number of Pokémon the models should fill data for at each single call
    max_chunk_size = 4

    # Generate training data and train model
    fine_tuner = FineTuner(max_chunk_size)
    asyncio.run(fine_tuner.generate_csv(all_pokemon_list))

    # Evaluate different models
    # evaluator = PokemonEvaluator(max_chunk_size)
    # df = asyncio.run(evaluator.evaluate(all_pokemon_list))
