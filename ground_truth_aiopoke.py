import json
import random
from typing import List, Tuple

import pandas as pd
import aiopoke
import aiohttp
from sentence_transformers import SentenceTransformer, util

from prompt import POKEMON_COLUMNS, dataframe_to_table_string


class PokemonGroundTruth:
    def __init__(self):
        self.similarity_model = None

    def _load_similarity_model(self):
        self.similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def _unload_similarity_model(self):
        self.similarity_model = None

    @staticmethod
    def random_chunk_generator(data, max_chunk_size):
        """
        Generate chunks of data from random indices with sizes up to a maximum chunk size.

        :param data: The list to chunk.
        :param max_chunk_size: The maximum size of any chunk.
        :return: A generator yielding chunks of the data from random indices.
        """

        remaining_indices = list(range(len(data)))
        while remaining_indices:
            chunk_size = min(random.randint(1, max_chunk_size), len(remaining_indices))
            selected_indices = random.sample(remaining_indices, chunk_size)
            yield [data[i] for i in selected_indices]

            # Removing the selected indices from the remaining indices list
            for i in selected_indices:
                remaining_indices.remove(i)

    async def generate_data(self, pokemon_names_path, max_pokemons=None):
        data = {"Input": [], "Output": []}

        # all_pokemon_list contains all 947 Pokémon names
        with open(pokemon_names_path) as pokemons_file:
            all_pokemon_list = json.load(pokemons_file)
        # max number of Pokémon the models should fill data for at each single call
        max_chunk_size = 4

        if max_pokemons:
            all_pokemon_list = all_pokemon_list[:max_pokemons]

        for chunk in self.random_chunk_generator(all_pokemon_list, max_chunk_size):
            input_str = ", ".join([shard.capitalize() for shard in chunk])
            data["Input"].append(input_str)

            ground_df = await self.fetch_pokemon_data(chunk)
            data["Output"].append(dataframe_to_table_string(ground_df))

            print(f"Appended Pokemon chuck: {chunk}")

        return pd.DataFrame(data)

    @staticmethod
    async def fetch_pokemon_data(pokemon_list: List[str]):
        async with aiohttp.ClientSession() as session:
            client = aiopoke.AiopokeClient(session=session)

            rows = []
            for pokemon_name in pokemon_list:
                pokemon_name = pokemon_name.lower()
                pokemon = await client.get_pokemon(pokemon_name)
                species = await client.get_pokemon_species(pokemon_name)

                # Fetching the evolution
                evolution_chain = await client.get_evolution_chain(
                    species.evolution_chain.id
                )
                current_evolution = evolution_chain.chain
                next_evolution = None

                # Traversing the evolution chain to find the next evolution
                while current_evolution:
                    if current_evolution.species.name == pokemon_name:
                        if current_evolution.evolves_to:
                            next_evolution = current_evolution.evolves_to[
                                0
                            ].species.name.capitalize()
                        break
                    if not current_evolution.evolves_to:
                        break
                    current_evolution = current_evolution.evolves_to[0]

                # Extract data
                name = pokemon.name.capitalize()
                types = ", ".join([t.type.name.capitalize() for t in pokemon.types])
                hp = pokemon.stats[0].base_stat
                attack = pokemon.stats[1].base_stat
                defense = pokemon.stats[2].base_stat
                special_attack = pokemon.stats[3].base_stat
                special_defense = pokemon.stats[4].base_stat
                speed = pokemon.stats[5].base_stat
                evolution = next_evolution or "None"

                # Combine all unique flavor texts into one description
                flavor_texts = set(
                    entry.flavor_text.replace("\n", " ").replace("\f", " ")
                    for entry in species.flavor_text_entries
                    if entry.language.name == "en"
                )
                description = " ".join(flavor_texts)

                rows.append(
                    [
                        name,
                        types,
                        hp,
                        attack,
                        defense,
                        special_attack,
                        special_defense,
                        speed,
                        evolution,
                        description,
                    ]
                )

            await client.close()

        return pd.DataFrame(
            rows,
            columns=POKEMON_COLUMNS,
        )

    @staticmethod
    def compute_type_score(gt_types: set, model_types: set) -> float:
        intersection = gt_types.intersection(model_types)
        union = gt_types.union(model_types)

        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity

    @staticmethod
    def weighted_score(
        name_accuracy,
        type_accuracy,
        hp_mae,
        attack_mae,
        defense_mae,
        special_attack_mae,
        special_defense_mae,
        speed_mae,
        evolution_accuracy,
        description_similarity,
    ):
        # Combine scores
        w_name = 0.1
        w_type = 0.1
        w_hp = 0.1
        w_attack = 0.1
        w_defense = 0.1
        w_special_attack = 0.1
        w_special_defense = 0.1
        w_speed = 0.1
        w_evolution = 0.1
        w_description = 0.1

        return (
            w_name * name_accuracy
            + w_type * type_accuracy
            + w_hp * (1 - hp_mae / 100)
            + w_attack * (1 - attack_mae / 100)
            + w_defense * (1 - defense_mae / 100)
            + w_special_attack * (1 - special_attack_mae / 100)
            + w_special_defense * (1 - special_defense_mae / 100)
            + w_speed * (1 - speed_mae / 100)
            + w_evolution * evolution_accuracy
            + w_description * description_similarity
        )

    async def score(self, model_df: pd.DataFrame) -> Tuple[float, float]:
        pokemon_list = list(model_df["Pokémon"])
        ground_truth = await self.fetch_pokemon_data(pokemon_list)

        # Pokémon Name comparison
        name_accuracy = sum(ground_truth["Pokémon"] == model_df["Pokémon"]) / len(
            ground_truth
        )

        # Compute type scores for all Pokémon
        type_scores = []
        for gt_type, model_type in zip(
            ground_truth["Type"].str.split(","), model_df["Type"].str.split(",|/")
        ):
            gt_type_set = set([t.strip().capitalize() for t in gt_type])
            model_type_set = set([t.strip().capitalize() for t in model_type])
            type_scores.append(self.compute_type_score(gt_type_set, model_type_set))

        type_accuracy = sum(type_scores) / len(ground_truth)

        # HP, Attack, Defense comparisons
        hp_mae = abs(ground_truth["HP"] - model_df["HP"].astype(int)).mean()
        attack_mae = abs(ground_truth["Attack"] - model_df["Attack"].astype(int)).mean()
        defense_mae = abs(
            ground_truth["Defense"] - model_df["Defense"].astype(int)
        ).mean()
        special_attack_mae = abs(
            ground_truth["Special Attack"] - model_df["Special Attack"].astype(int)
        ).mean()
        special_defense_mae = abs(
            ground_truth["Special Defense"] - model_df["Special Defense"].astype(int)
        ).mean()
        speed_mae = abs(ground_truth["Speed"] - model_df["Speed"].astype(int)).mean()

        # Evolution comparison
        evolution_accuracy = sum(
            ground_truth["Evolution"] == model_df["Evolution"].replace("-", "None")
        ) / len(ground_truth)

        if not self.similarity_model:
            self._load_similarity_model()

        # Compute embeddings
        embeddings_ground_truth = self.similarity_model.encode(
            ground_truth["Description"].tolist(), convert_to_tensor=True
        )
        embeddings_generated = self.similarity_model.encode(
            model_df["Description"].tolist(), convert_to_tensor=True
        )

        # Compute cosine similarity
        description_similarity = (
            util.cos_sim(embeddings_ground_truth, embeddings_generated).diag().mean()
        )

        total_score = self.weighted_score(
            name_accuracy=name_accuracy,
            type_accuracy=type_accuracy,
            hp_mae=hp_mae,
            attack_mae=attack_mae,
            defense_mae=defense_mae,
            special_attack_mae=special_attack_mae,
            special_defense_mae=special_defense_mae,
            speed_mae=speed_mae,
            evolution_accuracy=evolution_accuracy,
            description_similarity=description_similarity,
        )

        # Yes / No scoring system
        name_accuracy = name_accuracy == 1
        type_accuracy = type_accuracy == 1
        hp_mae = (1 - (hp_mae == 0)) * 100
        attack_mae = (1 - (attack_mae == 0)) * 100
        defense_mae = (1 - (defense_mae == 0)) * 100
        special_attack_mae = (1 - (special_attack_mae == 0)) * 100
        special_defense_mae = (1 - (special_defense_mae == 0)) * 100
        speed_mae = (1 - (speed_mae == 0)) * 100
        evolution_accuracy = evolution_accuracy == 1

        total_yn_score = self.weighted_score(
            name_accuracy=name_accuracy,
            type_accuracy=type_accuracy,
            hp_mae=hp_mae,
            attack_mae=attack_mae,
            defense_mae=defense_mae,
            special_attack_mae=special_attack_mae,
            special_defense_mae=special_defense_mae,
            speed_mae=speed_mae,
            evolution_accuracy=evolution_accuracy,
            description_similarity=description_similarity,
        )

        return float(total_score), float(total_yn_score)
