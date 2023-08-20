from typing import Tuple
from pydantic import BaseModel

from utils import table_to_dataframe


class Generation(BaseModel):
    completion: str
    time: float
    completion_tokens: int


class BaseLLM:
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Generation:
        raise NotImplementedError

    def print_output(self, prompt="Tell me about AI", temperature=0.7, max_tokens=200):
        print(f"Prompt: {prompt}")
        print("*** Generate:")
        generation = self.generate(prompt, temperature, max_tokens)
        print(generation.completion)
        print(
            f"Output generated in {generation.time:.2f} seconds ({generation.completion_tokens / generation.time:.2f} tokens/s, {generation.completion_tokens} tokens)"
        )

    def get_table_from_generator(self, prompt: str, temperature: float = 1.0, max_tokens: int = 150):
        while True:
            # Generate a completion based on the given prompt
            generation = self.generate(prompt, temperature, max_tokens)
            print(
                f"Output generated in {generation.time:.2f} seconds ({generation.completion_tokens / generation.time:.2f} tokens/s, {generation.completion_tokens} tokens)"
            )

            try:
                # Try to convert the completion into a DataFrame
                df = table_to_dataframe(generation.completion)

                # If successful, return the DataFrame
                return df
            except Exception as e:
                # If not successful, print the error (optional) and continue the loop
                print(f"Failed due to: {e}. Retrying...")

