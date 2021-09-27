"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import matplotlib.pyplot as plt


def plot_histograms(X_dict: dict,
                    data_type: str,
                    species_list: list[str],
                    variables_list: list[str],
                    variables_dict: dict,
                    show_plots: bool,
                    save_plots: bool,
                    plot_file_ext: str) -> None:
    for species in species_list:
        for variable in variables_list:
            plt.clf()

            plt.hist([X_dict[species[0]][data_type][variable],
                      X_dict[species[1]][data_type][variable]],
                     label=[species[0], species[1]],
                     stacked=True)
            plt.title(f'Espèces : {", ".join(species)}\nVariable : {variable}')
            plt.xlabel(f'{variable}')
            plt.ylabel('Fréquence')
            plt.legend()

            if show_plots:
                plt.show()

            if save_plots:
                plt.savefig(f'plots/hist_'
                            f'{"_".join(species)}'
                            f'_{variables_dict[variable]}'
                            f'.{plot_file_ext}')


def plot_scatter_plots(X_dict: dict,
                       data_type: str,
                       species_list: list[str],
                       variables_list: list[str],
                       variables_dict: dict,
                       show_plots: bool,
                       save_plots: bool,
                       plot_file_ext: str) -> None:
    for species in species_list:
        for variables in variables_list:
            plt.clf()

            plt.scatter(X_dict[species[0]][data_type][variables[0]],
                        X_dict[species[0]][data_type][variables[1]],
                        label=species[0])
            plt.scatter(X_dict[species[1]][data_type][variables[0]],
                        X_dict[species[1]][data_type][variables[1]],
                        label=species[1])

            plt.title(f'Espèces : {", ".join(species)}\nVariables : {", ".join(variables)}')
            plt.xlabel(variables[0])
            plt.ylabel(variables[1])
            plt.legend()

            if show_plots:
                plt.show()

            if save_plots:
                plt.savefig(f'plots/scatter_'
                            f'{"_".join(species)}'
                            f'_{"_".join([variables_dict[variable] for variable in variables])}'
                            f'.{plot_file_ext}')
