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


def get_list_combinations(my_list: list) -> list:
    output = []

    for i in range(len(my_list) + 1):
        for comb in combinations(my_list, i):
            output.append(list(comb))

    output.remove([])

    return output


def calculate_intra_distance(X_dict: dict,
                             species: str,
                             variables_list: list[str],
                             distance_method: str,
                             calculation_method: str) -> None:

    data = X_dict[species]['data']
    data = data[variables_list]

    metrics = X_dict[species]['metrics']
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

    distance_max = max(distance_list)

    print(f'Species: {species}, Variables: {variables_list}, Method: {distance_method} = {distance_max:.4f}')


if __name__ == '__main__':
    # Loading data from the csv file
    iris_data = pd.read_csv('data/iris.csv')

    variables_list = list(iris_data.columns.values)[:-1]

    variables_list_combs = get_list_combinations(variables_list)

    X = iris_data[variables_list]
    Y = iris_data[['species']]

    species_list = pd.unique(Y['species']).tolist()

    distance_methods = ['euclidean', 'mahalanobis']

    X_dict = {}

    for species in species_list:
        X_dict[species] = {}
        X_dict[species]['data'] = X.loc[Y['species'] == species]
        X_dict[species]['metrics'] = X_dict[species]['data'].describe()

    # 1.a.
    print('=== Intra-Class Distances ===\n')

    for species in species_list:
        for distance_method in distance_methods:
            for variables_comb in variables_list_combs:
                calculate_intra_distance(X_dict=X_dict,
                                         species=species,
                                         variables_list=variables_comb,
                                         distance_method=distance_method,
                                         calculation_method='manual')




    # # Euclidean Intra-Class Distance
    # intra_dist_setosa_eucl = pairwise_distances(X_setosa, metric='euclidean')
    # intra_dist_setosa_eucl = intra_dist_setosa_eucl.mean()
    #
    # # Mahalanobic Intra-Class Distance
    # intra_dist_setosa_maha_1 = cdist(XA=X_setosa.mean(axis=0).values.reshape(1, -1),
    #                                  XB=X_setosa,
    #                                  metric='mahalanobis',
    #                                  VI=np.linalg.inv(np.cov(X_setosa.T)))
    #
    # intra_dist_setosa_maha_1 = intra_dist_setosa_maha_1.mean()
    #
    #
    #
    #
    #
    # # 1.b
    #
    # print(iris_data)
