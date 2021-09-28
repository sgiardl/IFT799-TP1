"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.distance_fct import calculate_distance, verify_class_separation
from src.list_utils import get_all_combinations, get_combinations_of_two
from src.plotting import plot_histograms, plot_scatter_plots
from src.processing import format_intra_class_df, format_inter_class_df, print_latex_table, save_df_to_csv
from src.constants import PATH_CSV, PATH_PLOTS, PATH_LATEX


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing inputs')

    parser.add_argument('-v', '--verbose', action='store', type=bool, default=False,
                        help='Choose to print all results in the console')

    parser.add_argument('-cm', '--calculation_method', action='store', type=str, default='manual',
                        choices=['manual', 'scipy'],
                        help='Choose to use either a manual or scipy implementation of '
                             'euclidean and mahalanobis distance calculations')

    parser.add_argument('-norm_pca', '--normalize_pca', action='store', type=bool, default=True,
                        help='Choose to apply Z-score normalization to the data used for PCA')

    parser.add_argument('-sh_plts', '--show_plots', action='store', type=bool, default=False,
                        help='Choose to show plots')

    parser.add_argument('-sv_plts', '--save_plots', action='store', type=bool, default=True,
                        help='Choose to save plots in the plots/ directory')

    parser.add_argument('-pfe', '--plot_file_ext', action='store', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Choose')

    parser.add_argument('-ltx', '--print_latex', action='store', type=bool, default=True,
                        help='Choose to print LaTeX code for results tables')

    parser.add_argument('-nd', '--num_decimals', action='store', type=int, default=4,
                        help='Choose the number of decimals for floating point values rounding')

    parser.add_argument('-sv_csv', '--save_csv', action='store', type=bool, default=True,
                        help='Choose to save the results DataFrames into CSV files')

    args = parser.parse_args()

    print('Arguments:')
    print(vars(args))

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

    print('\nCalculating intra-class distances...')

    intra_class_df = pd.DataFrame(columns=['species', 'variables', 'method', 'distance'])

    for species in species_list:
        for distance_method in distance_methods:
            for variables_comb in variables_list_all_combs:
                intra_dist = calculate_distance(X_dict=X_dict,
                                                species_from='',
                                                species_to=species,
                                                variables_list=variables_comb,
                                                distance_method=distance_method,
                                                calculation_method=args.calculation_method,
                                                verbose=args.verbose,
                                                number_of_decimals=args.num_decimals)

                intra_class_df = intra_class_df.append({'species': species,
                                                        'variables': variables_comb,
                                                        'method': distance_method,
                                                        'distance': intra_dist}, ignore_index=True)

    if args.save_csv:
        save_df_to_csv(intra_class_df, filename='intra_class_raw', path=PATH_CSV)

    if args.verbose:
        print(intra_class_df)

    print('\nCalculating inter-class distances...')

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
                                                   calculation_method=args.calculation_method,
                                                   verbose=args.verbose,
                                                   number_of_decimals=args.num_decimals)

                inter_dist = calculate_distance(X_dict=X_dict,
                                                species_from=species_comb[0],
                                                species_to=species_comb[1],
                                                variables_list=variables_comb,
                                                distance_method=distance_method,
                                                calculation_method=args.calculation_method,
                                                verbose=args.verbose,
                                                number_of_decimals=args.num_decimals)

                flag_well_separated = verify_class_separation(species_from=species_comb[0],
                                                              species_to=species_comb[1],
                                                              intra_distance_to=intra_dist_to,
                                                              inter_distance=inter_dist,
                                                              verbose=args.verbose)

                inter_class_df = inter_class_df.append({'species from': species_comb[0],
                                                        'species to': species_comb[1],
                                                        'variables': variables_comb,
                                                        'method': distance_method,
                                                        'distance': inter_dist,
                                                        'well separated': flag_well_separated}, ignore_index=True)

    if args.verbose:
        print(inter_class_df)

    if args.save_csv:
        save_df_to_csv(inter_class_df, filename='inter_class_raw', path=PATH_CSV)

    # 2.c

    pca_n_components = len(variables_list)

    pca_components_list = [f'Composante PCA {i}' for i in range(1, pca_n_components + 1)]
    pca_components_list_two_combs = get_combinations_of_two(pca_components_list, include_rev=False)

    variables_dict = {variable: str(i) for i, variable in enumerate(variables_list, start=1)}

    for i, pca_component in enumerate(pca_components_list, start=1):
        variables_dict[pca_component] = f'pca{i}'

    pca = PCA(n_components=pca_n_components)

    if args.normalize_pca:
        X_PCA = pd.DataFrame(pca.fit_transform(StandardScaler().fit_transform(X.values)), columns=pca_components_list)
    else:
        X_PCA = pd.DataFrame(pca.fit_transform(X.values), columns=pca_components_list)

    for species in species_list:
        X_dict[species]['data_transformed'] = X_PCA.loc[Y['species'] == species]

    print('\nProcessing results...')

    clean_intra_class_df = format_intra_class_df(df_input=intra_class_df,
                                                 variables_list=variables_list_all_combs,
                                                 variables_dict=variables_dict,
                                                 species_list=species_list,
                                                 distance_methods=distance_methods,
                                                 number_of_decimals=args.num_decimals)
    # if args.print_latex:
    #     print_latex_table(df=clean_intra_class_df, header='Intra-Class Table')

    if args.save_csv:
        save_df_to_csv(clean_intra_class_df, filename='intra_class_clean', path=PATH_CSV)

    clean_inter_class_df_dict = {'_'.join(species_list_comb):
                                     format_inter_class_df(df_input=inter_class_df,
                                                           variables_list=variables_list_all_combs,
                                                           variables_dict=variables_dict,
                                                           species_list=species_list_comb,
                                                           distance_methods=distance_methods,
                                                           number_of_decimals=args.num_decimals)
                                 for species_list_comb in species_list_combs}

    for key, val in clean_inter_class_df_dict.items():
        # if args.print_latex:
        #     print_latex_table(df=val, header=f'Inter-Class Table: {key}')

        if args.save_csv:
            save_df_to_csv(val, filename=f'inter_class_clean_{key}', path=PATH_CSV)

    if args.show_plots or args.save_plots:
        # 2.a
        print('\nCreating plots...')

        plot_histograms(X_dict=X_dict,
                        data_type='data',
                        species_list=species_list_combs,
                        variables_list=variables_list,
                        variables_dict=variables_dict,
                        show_plots=args.show_plots,
                        save_plots=args.save_plots,
                        plot_file_ext=args.plot_file_ext,
                        path=PATH_PLOTS)

        # 2.a + 2.c
        plot_histograms(X_dict=X_dict,
                        data_type='data_transformed',
                        species_list=species_list_combs,
                        variables_list=pca_components_list,
                        variables_dict=variables_dict,
                        show_plots=args.show_plots,
                        save_plots=args.save_plots,
                        plot_file_ext=args.plot_file_ext,
                        path=PATH_PLOTS)

        # 2.b
        plot_scatter_plots(X_dict=X_dict,
                           data_type='data',
                           species_list=species_list_combs,
                           variables_list=variables_list_two_combs,
                           variables_dict=variables_dict,
                           show_plots=args.show_plots,
                           save_plots=args.save_plots,
                           plot_file_ext=args.plot_file_ext,
                           path=PATH_PLOTS)

        # 2.b + 2.c
        plot_scatter_plots(X_dict=X_dict,
                           data_type='data_transformed',
                           species_list=species_list_combs,
                           variables_list=pca_components_list_two_combs,
                           variables_dict=variables_dict,
                           show_plots=args.show_plots,
                           save_plots=args.save_plots,
                           plot_file_ext=args.plot_file_ext,
                           path=PATH_PLOTS)

        print(f'Plots have been saved to: {PATH_PLOTS}')
