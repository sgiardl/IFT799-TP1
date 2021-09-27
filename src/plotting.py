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
                    variables_list: list) -> None:
    for species in species_list:
        for variable in variables_list:
            plt.hist([X_dict[species[0]][data_type][variable],
                      X_dict[species[1]][data_type][variable]],
                     label=[species[0], species[1]],
                     stacked=True)
            plt.title(f'Espèces : {species[0]} vs. {species[1]}\nVariable : {variable}')
            plt.xlabel('Valeur')
            plt.ylabel('Fréquence')
            plt.legend()
            plt.show()


def plot_scatter_plots(X_dict: dict,
                       data_type: str,
                       species_list: list[str],
                       variables_list: list) -> None:
    for species in species_list:
        for variables in variables_list:
            plt.scatter(X_dict[species[0]][data_type][variables[0]],
                        X_dict[species[0]][data_type][variables[1]],
                        label=species[0])
            plt.scatter(X_dict[species[1]][data_type][variables[0]],
                        X_dict[species[1]][data_type][variables[1]],
                        label=species[1])

            plt.title(f'Espèces : {species[0]} vs. {species[1]}\nVariables : {variables}')
            plt.xlabel(variables[0])
            plt.ylabel(variables[1])
            plt.legend()
            plt.show()
