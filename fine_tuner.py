import pandas as pd

from ground_truth_aiopoke import PokemonGroundTruth
from prompt import random_chunk_generator, dataframe_to_table_string


class FineTuner:
    def __init__(self, max_chunk_size: int, save_path="finetune_data.csv"):
        self.max_chunk_size = max_chunk_size
        self.save_path = save_path

    async def generate_csv(self, pokemon_list):
        ground_truth_evaluator = PokemonGroundTruth()

        data = {"Input": [], "Output": []}

        for chunk in random_chunk_generator(pokemon_list, self.max_chunk_size):
            input_str = ", ".join([shard.capitalize() for shard in chunk])
            data["Input"].append(input_str)

            ground_df = await ground_truth_evaluator.fetch_pokemon_data(chunk)
            await ground_truth_evaluator.score(ground_df)
            data["Output"].append(dataframe_to_table_string(ground_df))

            print(f"Append chuck: {chunk}")

        df = pd.DataFrame(data)
        df.to_csv(self.save_path)
        print(f"Data saved to {self.save_path}")
