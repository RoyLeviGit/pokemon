from typing import Tuple, Union, List, Dict
from pydantic import BaseModel

from prompt import (
    table_string_to_dataframe,
    messages_for_model,
    POKEMON_SYSTEM_MESSAGE,
    POKEMON_ASSISTANT_MESSAGE,
)


class Generation(BaseModel):
    messages: Union[List[Dict[str, str]], str, List[str]]
    completion: str
    time: float
    completion_tokens: int


class BaseLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Generation:
        raise NotImplementedError

    def get_table_from_generator(
        self,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        messages = messages_for_model(
            model=self.model_name,
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

        # Convert the completion into a DataFrame
        df = table_string_to_dataframe(
            full_assistant_message, rows=len(user_message.split(","))
        )

        return df
