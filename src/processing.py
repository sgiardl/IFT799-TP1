"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd

from src.list_utils import get_combinations_of_two


def format_intra_class_df(df_input: pd.DataFrame,
                          variables_list: list[list[str]],
                          variables_dict: dict,
                          species_list: list[str],
                          distance_methods: list[str],
                          number_of_decimals: int) -> pd.DataFrame:
    columns = ['variables']

    for species in species_list:
        for distance_method in distance_methods:
            columns.append(f'{species}/{distance_method}')

    df_output = pd.DataFrame(columns=columns)

    for variables in variables_list:
        row = {columns[0]: ', '.join([variables_dict[variable] for variable in variables])}

        for species in species_list:
            for distance_method in distance_methods:
                filtered_df = df_input[(df_input['species'] == species) &
                                       (df_input['method'] == distance_method)]
                filtered_df = filtered_df[filtered_df['variables'].apply(lambda x: x == variables)]

                row[f'{species}/{distance_method}'] = round(float(filtered_df['distance'].values), number_of_decimals)

        df_output = df_output.append(row, ignore_index=True)

    return df_output


def format_inter_class_df(df_input: pd.DataFrame,
                          variables_list: list[list[str]],
                          variables_dict: dict,
                          species_list: list[str],
                          distance_methods: list[str],
                          number_of_decimals: int) -> pd.DataFrame:

    species_list = get_combinations_of_two(species_list, include_rev=True)

    columns = ['variables', 'species from', 'species to']

    for distance_method in distance_methods:
        columns.append(f'distance/{distance_method}')
        columns.append(f'separated/{distance_method}')

    df_output = pd.DataFrame(columns=columns)

    for variables in variables_list:
        row = {columns[0]: ', '.join([variables_dict[variable] for variable in variables])}

        for species in species_list:
            row[columns[1]] = species[0]
            row[columns[2]] = species[1]

            for distance_method in distance_methods:
                filtered_df = df_input[(df_input['species from'] == species[0]) &
                                       (df_input['species to'] == species[1]) &
                                       (df_input['method'] == distance_method)]
                filtered_df = filtered_df[filtered_df['variables'].apply(lambda x: x == variables)]

                row[f'distance/{distance_method}'] = round(float(filtered_df['distance'].values), number_of_decimals)
                row[f'separated/{distance_method}'] = bool(filtered_df['well separated'].values)

            df_output = df_output.append(row, ignore_index=True)

    return df_output


def print_latex_table(df: pd.DataFrame, header: str) -> None:
    print('*' * 50)
    print(f'{header}: LaTeX Code Beginning')
    print('*' * 50)

    print(df.to_latex(index=False))

    print('*' * 50)
    print('LaTeX Code End')
    print('*' * 50)


def save_df_to_csv(df: pd.DataFrame,
                   filename: str,
                   path: str) -> None:
    file_path = f'{path}{filename}.csv'

    df.to_csv(file_path)

    print(f'Results table has been saved to: {file_path}')
