import argparse
import getopt
import math
import random
import sys
from itertools import combinations, chain

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import neighbors, datasets
from sklearn.preprocessing import LabelEncoder


def read_data(file):
    return pd.read_csv(file)


def plot_distribution(df):
    sns.pairplot(df)
    plt.show()


def usage():
    print(f"Usage: genderclassification_knn.py --train-file <train.csv> [-pd]")


def neighbours(train_x, train_y, k):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_x, train_y)
    return classifier


def find_best_subset(estimator, X, y, max_size=8, cv=5, print_progress=True):
    """
    Calculates the best model of up to max_size features of X.
    estimator must have a fit and score functions.
    X must be a DataFrame.
    Source of function: https://stackoverflow.com/a/50704252/6400551
    """

    n_features = X.shape[1]
    subsets = (combinations(range(n_features), k + 1)
               for k in range(min(n_features, max_size)))

    subsets_2 = (combinations(range(n_features), k + 1)
                 for k in range(min(n_features, max_size)))

    best_size_subset = []

    progress_percentage = 0
    progress = 0
    # total_combinations = sum(math.comb(n_features, size) for size in range(max_size + 1))
    total_combinations = 0

    for subsets_k in subsets_2:
        for subset in subsets_k:
            total_combinations += 1

    if print_progress:
        print(f"Looking through {total_combinations} combinations...")

    for subsets_k in subsets:  # for each list of subsets of the same size
        best_score = -np.inf
        best_subset = None

        for subset in subsets_k:
            if print_progress:
                progress += 1

                percentage = 100.0 * (progress / float(total_combinations))
                percentage_int = int(percentage)

                if percentage_int > progress_percentage:
                    progress_percentage = percentage_int
                    print(f"Progress: {progress_percentage}%")

            estimator.fit(X.iloc[:, list(subset)], y)
            # get the subset with the best score among subsets of the same size
            score = estimator.score(X.iloc[:, list(subset)], y)
            if score > best_score:
                best_score, best_subset = score, subset

        # to compare subsets of different sizes we must use CV
        # first store the best subset of each size
        best_size_subset.append(best_subset)

    # compare best subsets of each size
    best_score = -np.inf
    best_subset = None
    list_scores = []
    for subset in best_size_subset:
        score = cross_val_score(estimator, X.iloc[:, list(subset)], y, cv=cv).mean()
        list_scores.append(score)
        if score > best_score:
            best_score, best_subset = score, subset

    return best_subset, best_score, best_size_subset, list_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pd', '--plot-distribution', dest='pd', action='store_true')
    parser.add_argument('-t', '--train-file', dest='train', action='store')
    parser.add_argument('-l', '--label', dest='label', action='store')
    parser.add_argument('-s', '--split', dest='split', action='store', type=float, default=0.75)
    parser.add_argument('-r', '--random-state', dest='random', type=int, default=None)
    parser.add_argument('-k', '--k', dest='k', action='store', type=int, default=10)
    parser.add_argument('-f', '--feature-analysis', dest='features', action='store_true')
    parser.add_argument('-kmin', '--k-min', dest='kmin', action='store', type=int, default=None)
    parser.add_argument('-kmax', '--k-max', dest='kmax', action='store', type=int, default=None)
    args = parser.parse_args()

    if args.train is None:
        usage()
        return

    df = read_data(args.train)

    if args.pd:
        plot_distribution(df)

    df[args.label] = LabelEncoder().fit_transform(df.Lead.values)
    features = df.drop([args.label], axis=1)
    labels = df[args.label]
    state = random.Random().randint(0, 99999) if args.random is None else args.random

    train_x, test_x, train_y, test_y = train_test_split(features, labels,
                                                        train_size=args.split,
                                                        random_state=state)

    classifier = neighbours(train_x, train_y, args.k)
    score = classifier.score(test_x, test_y)

    print(f"k: {args.k} achieves score: {score}")

    if args.features:
        best_set, best_score, best_size_subset, list_scores = \
            find_best_subset(neighbors.KNeighborsClassifier(), train_x, train_y)

        print(f"Best subset: {best_set}")
        print(f"Best score: {best_score}")
        print(f"Best size subset: {best_size_subset}")
        print(f"List of scores: {list_scores}")

    k_min, k_max = args.kmin, args.kmax

    if k_min is not None and k_max is not None:
        k_range = range(k_min, k_max + 1)
        plot_x = []
        plot_y = []

        param_grid = dict(n_neighbors=k_range)

        for current_k in k_range:
            classifier = neighbours(train_x, train_y, current_k)
            score = classifier.score(test_x, test_y)

            plot_x.append(current_k)
            plot_y.append(score)

        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title("k-Nearest Neighbours: score vs k")

        sns.scatterplot(
            x=plot_x,
            y=plot_y
        )

        plt.show()


if __name__ == '__main__':
    main()
