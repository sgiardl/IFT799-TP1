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
        if species_from == '':
            covar_matrix = data.cov().values
        else:
            data_to = X_dict[species_to]['data']
            data_to = data_to[variables_list]
            covar_matrix = data_to.cov().values

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
