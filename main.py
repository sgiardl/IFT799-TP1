"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.distance_fct import calculate_distance, verify_class_separation
from src.list_utils import get_all_combinations, get_combinations_of_two
from src.plotting import plot_histograms, plot_scatter_plots


if __name__ == '__main__':
    # User Options
    verbose = False
    calculation_method = 'manual'  # choices = 'manual' or 'scipy'

    # Loading data from the csv file
    iris_data = pd.read_csv('data/iris.csv')

    variables_list = list(iris_data.columns.values)[:-1]
    variables_list_all_combs = get_all_combinations(variables_list)
    variables_list_two_combs = get_combinations_of_two(variables_list, include_rev=False)

    X = iris_data[variables_list]
    Y = iris_data[['species']]

    species_list = pd.unique(Y['species']).tolist()
    species_list_combs_with_rev = get_combinations_of_two(species_list, include_rev=True)
    species_list_combs = get_combinations_of_two(species_list, include_rev=False)

    distance_methods = ['euclidean', 'mahalanobis']

    X_dict = {}

    for species in species_list:
        X_dict[species] = {}
        X_dict[species]['data'] = X.loc[Y['species'] == species]
        X_dict[species]['metrics'] = X_dict[species]['data'].describe()

    # 1.a + 1.b

    print('=== Intra-Class Distances ===\n')

    intra_class_df = pd.DataFrame(columns=['species', 'variables', 'method', 'distance'])

    for species in species_list:
        for distance_method in distance_methods:
            for variables_comb in variables_list_all_combs:
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

    for species_comb in species_list_combs_with_rev:
        for distance_method in distance_methods:
            for variables_comb in variables_list_all_combs:
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

    # 2.c

    pca_n_components = len(variables_list)

    pca_components_list = [f'PCA Component {i + 1}' for i in range(pca_n_components)]
    pca_components_list_two_combs = get_combinations_of_two(pca_components_list, include_rev=False)

    pca = PCA(n_components=pca_n_components)
    X_PCA = pd.DataFrame(pca.fit_transform(StandardScaler().fit_transform(X.values)), columns=pca_components_list)

    for species in species_list:
        X_dict[species]['data_transformed'] = X_PCA.loc[Y['species'] == species]

    # 2.a

    plot_histograms(X_dict=X_dict,
                    data_type='data',
                    species_list=species_list_combs,
                    variables_list=variables_list)

    # 2.a + 2.c

    plot_histograms(X_dict=X_dict,
                    data_type='data_transformed',
                    species_list=species_list_combs,
                    variables_list=pca_components_list)

    # 2.b

    plot_scatter_plots(X_dict=X_dict,
                       data_type='data',
                       species_list=species_list_combs,
                       variables_list=variables_list_two_combs)

    # 2.b + 2.c

    plot_scatter_plots(X_dict=X_dict,
                       data_type='data_transformed',
                       species_list=species_list_combs,
                       variables_list=pca_components_list_two_combs)
