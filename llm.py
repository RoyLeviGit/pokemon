from typing import Tuple, Union, List, Dict
from pydantic import BaseModel

from prompt import (
    table_string_to_dataframe,
    messages_for_model,
    POKEMON_SYSTEM_MESSAGE,
    POKEMON_ASSISTANT_MESSAGE,
    SUMMARIZER_SYSTEM_MESSAGE,
    SUMMARIZER_ASSISTANT_MESSAGE,
)


class Generation(BaseModel):
    messages: Union[List[Dict[str, str]], str, List[str]]
    completion: str
    time: float
    completion_tokens: int


class BaseLLM:
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Generation:
        raise NotImplementedError

    def get_table_from_generator(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 5,
    ):
        retries = 0
        while retries < max_retries:
            messages = messages_for_model(
                model=self.model,
                system_message=POKEMON_SYSTEM_MESSAGE,
                user_message=user_message,
                assistant_message=POKEMON_ASSISTANT_MESSAGE,
            )
            # Generate a completion based on the given messages
            generation = self.generate(
                messages,
                temperature,
                max_tokens,
            )
            print(f"Messages:\n{generation.messages}")
            full_assistant_message = POKEMON_ASSISTANT_MESSAGE + generation.completion
            print(f"Full assistant message:\n{full_assistant_message}")
            print(
                f"Output generated in {generation.time:.2f} seconds ({generation.completion_tokens / generation.time:.2f} tokens/s, {generation.completion_tokens} tokens)"
            )

            try:
                # Try to convert the completion into a DataFrame
                df = table_string_to_dataframe(
                    full_assistant_message, rows=len(user_message.split(","))
                )

                # If successful, return the DataFrame
                return df
            except Exception as e:
                # If not successful, print the error (optional) and continue the loop
                print(f"Failed due to: {e}. Retrying...")

        # Raise an exception or return a default value after exceeding the max retries
        raise Exception(
            f"Failed to generate a valid table after {max_retries} attempts."
        )

    def get_summary_from_generator(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        messages = messages_for_model(
            model=self.model,
            system_message=SUMMARIZER_SYSTEM_MESSAGE,
            user_message=user_message,
            assistant_message=SUMMARIZER_ASSISTANT_MESSAGE,
        )
        # Generate a completion based on the given messages
        generation = self.generate(
            messages,
            temperature,
            max_tokens,
        )
        print(f"Messages:\n{generation.messages}")
        full_assistant_message = SUMMARIZER_ASSISTANT_MESSAGE + generation.completion
        print(f"Full assistant message:\n{full_assistant_message}")
        print(
            f"Output generated in {generation.time:.2f} seconds ({generation.completion_tokens / generation.time:.2f} tokens/s, {generation.completion_tokens} tokens)"
        )

        return generation.completion
