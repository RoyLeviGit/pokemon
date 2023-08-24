import asyncio
import json

import pandas as pd
from dotenv import load_dotenv

from evaluator import PokemonEvaluator
from fine_tuner import FineTuner
from ground_truth_aiopoke import PokemonGroundTruth
from llama2_exllama import Llama2ExLlama
from gpt4_openai import GPT4OpenAI
from prompt import random_chunk_generator

if __name__ == "__main__":
    load_dotenv()

    # all_pokemon_list contains all 947 Pokémon names
    # with open("data/pokemon_names.json") as pokemons_file:
    #     all_pokemon_list = json.load(pokemons_file)
    # max number of Pokémon the models should fill data for at each single call
    # max_chunk_size = 4

    # Generate training data and train model
    fine_tuner = FineTuner()
    # asyncio.run(fine_tuner.generate_csv(all_pokemon_list, max_chunk_size))
    fine_tuner.format_csv_for_train()

    # Evaluate different models
    # evaluator = PokemonEvaluator()

    # Llama2 random evaluation
    # llm = Llama2ExLlama()
    # asyncio.run(
    #     evaluator.evaluate(
    #         chunks=random_chunk_generator(all_pokemon_list, max_chunk_size), llm=llm
    #     )
    # )

    # GPT4 random evaluation
    # llm = GPT4OpenAI()
    # asyncio.run(
    #     evaluator.evaluate(
    #         chunks=random_chunk_generator(all_pokemon_list, max_chunk_size), llm=llm
    #     )
    # )

    # Get previous result inputs
    # with open("data/results_Llama2ExLlama.csv") as results_file:
    #     results = pd.read_csv(results_file)
    #     chunks = [
    #         [i.strip() for i in r_input.split(",")]
    #         for r_input in results["Input"].tolist()
    #     ]

    # GPT4 repeated inputs evaluation
    # llm = GPT4OpenAI()
    # asyncio.run(evaluator.evaluate(chunks=chunks, llm=llm))
