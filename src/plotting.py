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
    """
    Commentaire dans le TP :
        Pour faire un histogramme, vous devez choisir vous-
        même la largeur de chaque « bin », alors que la largeur affecte la qualité visuelle d’un
        histogramme. Quand vous affichez deux histogrammes sur une même figure, il vaut
        mieux afficher un histogramme « par-dessus » l’autre.

        SGL : ??? matplotlib gère déjà bien les bins
    """

    for species in species_list:
        for variable in variables_list:
            plt.hist([X_dict[species[0]][data_type][variable],
                      X_dict[species[1]][data_type][variable]],
                     label=[species[0], species[1]])
            plt.title(f'{species[0]} vs. {species[1]}, Variable: {variable}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
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

            plt.title(f'{species[0]} vs. {species[1]}, Variables: {variables}')
            plt.xlabel(variables[0])
            plt.ylabel(variables[1])
            plt.legend()
            plt.show()
