from typing import Union, List, Dict

import pandas as pd

POKEMON_COLUMNS = [
    "Pokémon",
    "Type",
    "HP",
    "Attack",
    "Defense",
    "Special Attack",
    "Special Defense",
    "Speed",
    "Evolution",
    "Description",
]

POKEMON_SYSTEM_MESSAGE = (
    """Given a list of Pokémon names, write a table with these headers:
headers = [\""""
    + '", "'.join(POKEMON_COLUMNS)
    + '"]'
)

POKEMON_ASSISTANT_MESSAGE = "| " + " | ".join(POKEMON_COLUMNS) + " |\n"

SUMMARIZER_SYSTEM_MESSAGE = """Given a piece of text, I you want to summarize.
The goal is to generate a concise summary that captures the main points of the text."""

SUMMARIZER_ASSISTANT_MESSAGE = """Sure! Here's a summary of the text you provided:
"""

LLAMA2_PROMPT_TEMPLATE = """[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST] {assistant_message}"""


def messages_for_model(
    model: str, system_message: str, user_message: str, assistant_message: str
) -> Union[List[Dict[str, str]], str, List[str]]:
    """
    Generate messages formatted for a specific model.

    Parameters:
    - model (str): The name of the model, e.g., "gpt4", "llama2", etc.
    - system_message (str): The system message content.
    - user_message (str): The user message content.
    - assistant_message (str): The assistant message content.

    Returns:
    - Union[List[Dict[str, str]], str, List[str]]: Formatted messages for the model.
    """

    model_templates = {
        "gpt": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "llama": LLAMA2_PROMPT_TEMPLATE.format(
            system_message=system_message,
            user_message=user_message,
            assistant_message=assistant_message,
        ),
        "ground_truth": [p.strip() for p in user_message.split(",")],
    }

    for model_template_key in model_templates.keys():
        if model_template_key in model:
            return model_templates[model_template_key]

    raise NotImplementedError(f"Model '{model}' not supported.")


def table_string_to_dataframe(s: str, rows: int = None) -> pd.DataFrame:
    # Split the input string into lines
    lines = s.strip().split("\n")

    # Identify the start and end of the table based on the presence of '|'
    table_lines = [
        line
        for line in lines
        if "|" in line and "---" not in line and ":-:" not in line
    ]
    if rows:
        table_lines = table_lines[: rows + 1]

    # Extract the headers and data separately
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    data = [
        [cell.strip() for cell in row.strip("|").split("|")] for row in table_lines[1:]
    ]

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=headers)

    return df


def dataframe_to_table_string(df: pd.DataFrame) -> str:
    # Convert headers to a list of strings
    headers = df.columns.tolist()

    # Convert the DataFrame rows to a list of lists of strings
    data = df.values.tolist()

    # Create a string for headers
    header_str = " | ".join(headers)

    # Create a separator string based on the headers
    separator = "| --- " * len(headers) + "|"

    # Convert the data to string and join them
    data_strs = [" | ".join(map(str, row)) for row in data]

    # Combine everything together
    table_str = f"| {header_str} |\n{separator}\n" + "\n".join(
        [f"| {row} |" for row in data_strs]
    )

    return table_str
