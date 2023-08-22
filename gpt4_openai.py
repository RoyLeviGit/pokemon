import os
import time

import openai

from llm import BaseLLM, Generation


class GPT4OpenAI(BaseLLM):
    def __init__(self, model="gpt-4"):
        super().__init__(model)

        if "OPENAI_API_KEY" not in os.environ:
            raise openai.error.AuthenticationError("OPENAI_API_KEY not in os.environ")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, messages, temperature, max_tokens):
        t0 = time.time()
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        t1 = time.time()

        return Generation(
            messages=messages,
            completion=completion.choices[0].message["content"],
            time=t1 - t0,
            completion_tokens=completion.usage["completion_tokens"],
        )
