import asyncio

from dotenv import load_dotenv

from ground_truth_aiopoke import PokemonGroundTruth
from llama2_exllama import Llama2ExLlama
from gpt4_openai import GPT4OpenAI
from prompt import dataframe_to_table_string

if __name__ == "__main__":
    load_dotenv()

    pokemon_list = ["Bulbasaur", "Ivysaur", "Venusaur"]

    ground_truth_evaluator = PokemonGroundTruth()
    ground_df = asyncio.run(ground_truth_evaluator.fetch_pokemon_data(pokemon_list))
    print(f"Ground dataframe:\n{dataframe_to_table_string(ground_df)}")
    score = asyncio.run(ground_truth_evaluator.score(ground_df))
    print(f"Ground score: {score}")

    llms = [GPT4OpenAI(), Llama2ExLlama()]
    for llm in llms:
        df = llm.get_table_from_generator(", ".join(pokemon_list))
        score = asyncio.run(ground_truth_evaluator.score(df))
        print(f"{llm.__class__.__name__} score: {score}")
