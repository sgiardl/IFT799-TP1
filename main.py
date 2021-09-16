'''
IFT799 - Science des données
TP1
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
'''

import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

if __name__ == '__main__':
    # Loading data from the csv file
    iris_data = pd.read_csv('data/iris.csv')

    X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    Y = iris_data[['species']]

    X_setosa = X.loc[Y['species'] == 'setosa']
    X_versicolor = X.loc[Y['species'] == 'versicolor']
    X_virginica = X.loc[Y['species'] == 'virginica']

    # 1.a
    intra_distance_setosa = pairwise_distances(X_setosa, metric='euclidean')

    # 1.b

    print(iris_data)
