"""
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from itertools import combinations


def get_all_combinations(my_list: list) -> list:
    all_combs = []

    for i in range(len(my_list) + 1):
        for comb in combinations(my_list, i):
            all_combs.append(list(comb))

    all_combs.remove([])

    return all_combs


def get_combinations_of_two(my_list: list, include_rev: bool) -> list:
    combs_of_two = []

    for comb in combinations(my_list, 2):
        combs_of_two.append(list(comb))

        if include_rev:
            combs_of_two.append(list(comb))

    if include_rev:
        for i in range(len(combs_of_two)):
            if i % 2 != 0:
                combs_of_two[i].reverse()

    return combs_of_two
