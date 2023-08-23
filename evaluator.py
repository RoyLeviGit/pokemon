import pandas as pd
import os
import asyncio

from gpt4_openai import GPT4OpenAI
from ground_truth_aiopoke import PokemonGroundTruth
from llama2_exllama import Llama2ExLlama
from prompt import dataframe_to_table_string, random_chunk_generator


class PokemonEvaluator:
    def __init__(self, max_chunk_size, save_path):
        self.max_chunk_size = max_chunk_size
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            # Initialize the CSV file if it doesn't exist
            df = pd.DataFrame(
                columns=[
                    "Input",
                    "Ground truth",
                    "GPT4OpenAI",
                    "Llama2ExLlama",
                    "GPT4OpenAI score",
                    "Llama2ExLlama score",
                ]
            )
            df.to_csv(self.save_path, index=False)

    def _load_processed(self):
        # Load processed data
        return pd.read_csv(self.save_path)["Input"].tolist()

    def _save_chunk_results(self, results):
        # Save intermediate results
        df = pd.read_csv(self.save_path)
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        df.to_csv(self.save_path, index=False)

    async def evaluate(self, pokemon_list):
        llms = [Llama2ExLlama(), GPT4OpenAI()]
        ground_truth_evaluator = PokemonGroundTruth()

        processed = self._load_processed()

        for chunk in random_chunk_generator(pokemon_list, self.max_chunk_size):
            input_str = ", ".join([shard.capitalize() for shard in chunk])
            if input_str in processed:
                # Skip if already processed
                continue

            results = {"Input": input_str}

            skip_chunk = False
            for llm in llms:
                try:
                    df = llm.get_table_from_generator(input_str)
                    results[f"{llm.__class__.__name__}"] = dataframe_to_table_string(df)
                    results[
                        f"{llm.__class__.__name__} score"
                    ] = await ground_truth_evaluator.score(df)
                except ValueError:
                    print(f'Skipping chunk for input: "{input_str}"')
                    skip_chunk = True
                    break

            if not skip_chunk:
                # Fetch ground truth data
                ground_df = await ground_truth_evaluator.fetch_pokemon_data(chunk)
                results["Ground truth"] = dataframe_to_table_string(ground_df)

                # Save results for this chunk
                self._save_chunk_results([results])

        return pd.read_csv(self.save_path)
