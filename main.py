import os

from dotenv import load_dotenv

from llama_2_autogptq import Llama2GPTQ
from gpt4_openai import GPT4OpenAI


if __name__ == "__main__":
    load_dotenv()

    df = GPT4OpenAI(os.getenv("OPENAI_API_KEY")).get_table_from_generator("write a table with 5 pokemon and their attributes")
    print(df)
    df = Llama2GPTQ().get_table_from_generator("write a table with 5 pokemon and their attributes")
    print(df)

