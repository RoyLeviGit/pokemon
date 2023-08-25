import pandas as pd
import os

from ground_truth_aiopoke import PokemonGroundTruth
from prompt import dataframe_to_table_string, random_chunk_generator


class PokemonEvaluator:
    @staticmethod
    def _load_processed(model_name, save_path):
        if not os.path.exists(save_path):
            # Initialize the CSV file if it doesn't exist
            df = pd.DataFrame(
                columns=[
                    "Input",
                    model_name,
                    f"{model_name} score",
                    f"{model_name} yn score",
                ]
            )
            df.to_csv(save_path, index=False)

        # Load processed data
        return pd.read_csv(save_path)["Input"].tolist()

    @staticmethod
    def _save_chunk_results(results, save_path):
        # Save intermediate results
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        df.to_csv(save_path, index=False)

    async def evaluate(self, chunks, llm):
        save_path = "data/results_train_" + llm.__class__.__name__ + ".csv"
        ground_truth_evaluator = PokemonGroundTruth()

        processed = self._load_processed(
            model_name=llm.__class__.__name__, save_path=save_path
        )

        num_of_shards = len([shard for chunk in chunks for shard in chunk])
        remaining_shards = num_of_shards

        for chunk in chunks:
            print(
                f"Progress({num_of_shards - remaining_shards}/{num_of_shards}):{(num_of_shards - remaining_shards) / num_of_shards}"
            )
            remaining_shards -= len(chunk)

            input_str = ", ".join([shard.capitalize() for shard in chunk])
            if input_str in processed:
                # Skip if already processed
                continue

            results = {"Input": input_str}

            try:
                df = llm.get_table_from_generator(input_str)
                results[f"{llm.__class__.__name__}"] = dataframe_to_table_string(df)
                score, yn_score = await ground_truth_evaluator.score(df)
                results[f"{llm.__class__.__name__} score"] = score
                results[f"{llm.__class__.__name__} yn score"] = yn_score
            except:
                print(f'Skipping chunk: "{chunk}"')
                continue

            # Save results for this chunk
            self._save_chunk_results([results], save_path)

        return pd.read_csv(save_path)
