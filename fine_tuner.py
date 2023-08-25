import pandas as pd

from ground_truth_aiopoke import PokemonGroundTruth
from prompt import (
    random_chunk_generator,
    dataframe_to_table_string,
    LLAMA2_PROMPT_TEMPLATE,
    POKEMON_SYSTEM_MESSAGE,
    messages_for_model,
)


class FineTuner:
    def __init__(self, save_path="data/finetune_data.csv"):
        self.save_path = save_path

    async def generate_csv(self, pokemon_list, max_chunk_size):
        ground_truth_evaluator = PokemonGroundTruth()

        data = {"Input": [], "Output": []}

        for chunk in random_chunk_generator(pokemon_list, max_chunk_size):
            input_str = ", ".join([shard.capitalize() for shard in chunk])
            data["Input"].append(input_str)

            ground_df = await ground_truth_evaluator.fetch_pokemon_data(chunk)
            await ground_truth_evaluator.score(ground_df)
            data["Output"].append(dataframe_to_table_string(ground_df))

            print(f"Append chuck: {chunk}")

        df = pd.DataFrame(data)
        df.to_csv(self.save_path)
        print(f"Data saved to {self.save_path}")

    def format_csv_for_train(self):
        with open(self.save_path, encoding="utf8") as data_file:
            data = pd.read_csv(data_file)

        train_data = pd.DataFrame()
        train_data["text"] = data.apply(
            lambda row: messages_for_model(
                model="llama-2",
                system_message=POKEMON_SYSTEM_MESSAGE,
                user_message=row["Input"],
                assistant_message=row["Output"],
            ),
            axis=1,
        )

        train_data.to_csv("data/train.csv")
        print(f"Train data saved to data/train.csv")
