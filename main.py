"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
from itertools import combinations


def get_variables_combinations(variables: list) -> list:
    variables_combs = []

    for i in range(len(variables) + 1):
        for comb in combinations(variables, i):
            variables_combs.append(list(comb))

    variables_combs.remove([])

    return variables_combs


def get_species_combinations(species: list) -> list:
    species_combs = []

    for comb in combinations(species, 2):
        species_combs.append(list(comb))
        species_combs.append(list(comb))

    for i in range(len(species_combs)):
        if i % 2 != 0:
            species_combs[i].reverse()

    return species_combs


def calculate_distance(X_dict: dict,
                       species_from: str,
                       species_to: str,
                       variables_list: list[str],
                       distance_method: str,
                       calculation_method: str,
                       verbose: bool = True,
                       number_of_decimals: int = 4) -> float:

    # Intra-class distance if species_from == '', else directional inter-class distance
    data = X_dict[species_to]['data'] if species_from == '' else X_dict[species_from]['data']
    data = data[variables_list]

    metrics = X_dict[species_to]['metrics']
    metrics = metrics[variables_list]

    distance_list = []

    if distance_method == 'euclidean':
        # Using scipy
        if calculation_method == 'scipy':
            for _, row in data.iterrows():
                distance_list.append(distance.euclidean(row, metrics.loc['mean']))

        # Manually
        elif calculation_method == 'manual':
            for _, row in data.iterrows():
                point_list = []

                for name, col in row.iteritems():
                    point_list.append((col - metrics.loc['mean'][name]) ** 2)

                distance_list.append(sum(point_list) ** 0.5)

    elif distance_method == 'mahalanobis':
        covar_matrix = data.cov().values
        inv_covar_matrix = np.linalg.inv(covar_matrix)

        # Using scipy
        if calculation_method == 'scipy':
            for _, row in data.iterrows():
                distance_list.append(distance.mahalanobis(row, metrics.loc['mean'], inv_covar_matrix))

        # Manually
        elif calculation_method == 'manual':
            q = metrics.loc['mean']

            for _, p in data.iterrows():
                distance_list.append(np.array(p - q).T.dot(inv_covar_matrix).dot(np.array(p - q)) ** 0.5)

    else:
        raise ValueError('Wrong distance method entered! Valid choices are: '
                         '["euclidean", "mahalanobis"]')

    # Intra-class distance
    if species_from == '':
        distance_max = max(distance_list)

        if verbose:
            print(f'Species: {species_to}, '
                  f'Variables: {variables_list}, Method: {distance_method} '
                  f'= {distance_max:.{number_of_decimals}f}')

        return distance_max

    # Inter-class distance
    else:
        distance_min = min(distance_list)

        if verbose:
            print(f'Species From: {species_from}, Species To: {species_to}, '
                  f'Variables: {variables_list}, Method: {distance_method} '
                  f'= {distance_min:.{number_of_decimals}f}')

        return distance_min


def verify_class_separation(species_from: str,
                            species_to: str,
                            intra_distance_to: float,
                            inter_distance: float,
                            verbose: bool = True) -> bool:
    if intra_distance_to < inter_distance:
        if verbose:
            print(f'The species {species_from} and {species_to} are well separated.')

        return True
    else:
        if verbose:
            print(f'The species {species_from} and {species_to} are NOT well separated.')

        return False


if __name__ == '__main__':
    verbose = False

    # Loading data from the csv file
    iris_data = pd.read_csv('data/iris.csv')

    variables_list = list(iris_data.columns.values)[:-1]
    variables_list_combs = get_variables_combinations(variables_list)

    X = iris_data[variables_list]
    Y = iris_data[['species']]

    species_list = pd.unique(Y['species']).tolist()
    species_list_combs = get_species_combinations(species_list)

    distance_methods = ['euclidean', 'mahalanobis']

    X_dict = {}

    for species in species_list:
        X_dict[species] = {}
        X_dict[species]['data'] = X.loc[Y['species'] == species]
        X_dict[species]['metrics'] = X_dict[species]['data'].describe()

    print('=== Intra-Class Distances ===\n')

    intra_class_df = pd.DataFrame(columns=['species', 'variables', 'method', 'distance'])

    for species in species_list:
        for distance_method in distance_methods:
            for variables_comb in variables_list_combs:
                intra_dist = calculate_distance(X_dict=X_dict,
                                                species_from='',
                                                species_to=species,
                                                variables_list=variables_comb,
                                                distance_method=distance_method,
                                                calculation_method='manual',
                                                verbose=verbose)

                intra_class_df = intra_class_df.append({'species': species,
                                                        'variables': variables_comb,
                                                        'method': distance_method,
                                                        'distance': intra_dist}, ignore_index=True)

    print(intra_class_df)

    print('\n=== Inter-Class Distances ===\n')

    inter_class_df = pd.DataFrame(columns=['species from', 'species to', 'variables',
                                           'method', 'distance', 'well separated'])

    for species_comb in species_list_combs:
        for distance_method in distance_methods:
            for variables_comb in variables_list_combs:
                intra_dist_to = calculate_distance(X_dict=X_dict,
                                                   species_from='',
                                                   species_to=species_comb[1],
                                                   variables_list=variables_comb,
                                                   distance_method=distance_method,
                                                   calculation_method='manual',
                                                   verbose=verbose)

                inter_dist = calculate_distance(X_dict=X_dict,
                                                species_from=species_comb[0],
                                                species_to=species_comb[1],
                                                variables_list=variables_comb,
                                                distance_method=distance_method,
                                                calculation_method='manual',
                                                verbose=verbose)

                flag_well_separated = verify_class_separation(species_from=species_comb[0],
                                                              species_to=species_comb[1],
                                                              intra_distance_to=intra_dist_to,
                                                              inter_distance=inter_dist,
                                                              verbose=verbose)

                inter_class_df = inter_class_df.append({'species from': species_comb[0],
                                                        'species to': species_comb[1],
                                                        'variables': variables_comb,
                                                        'method': distance_method,
                                                        'distance': inter_dist,
                                                        'well separated': flag_well_separated}, ignore_index=True)

    print(inter_class_df)



