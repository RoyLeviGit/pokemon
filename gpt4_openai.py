import time

import openai

from llm import BaseLLM, Generation


class GPT4OpenAI(BaseLLM):
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def generate(self, prompt, temperature, max_tokens):
        t0 = time.time()
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        t1 = time.time()

        return Generation(
            completion=completion.choices[0].message["content"],
            time=t1 - t0,
            completion_tokens=completion.usage["completion_tokens"],
        )
