import pandas as pd


def table_to_dataframe(s: str) -> pd.DataFrame:
    # Split the input string into lines
    lines = s.strip().split('\n')

    # Identify the start and end of the table based on the presence of '|'
    table_lines = [line for line in lines if '|' in line and '---' not in line]

    # Extract the headers and data separately
    headers = [cell.strip() for cell in table_lines[0].strip('|').split('|')]
    data = [[cell.strip() for cell in row.strip('|').split('|')] for row in table_lines[1:]]

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=headers)

    return df
