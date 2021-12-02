import argparse
import getopt
import random
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pd', '--plot-distribution', dest='pd', action='store_true')
    parser.add_argument('-t', '--train-file', dest='train', action='store')
    parser.add_argument('-l', '--label', dest='label', action='store')
    parser.add_argument('-s', '--split', dest='split', action='store', type=float, default=0.75)
    parser.add_argument('-r', '--random-state', dest='random', type=int, default=None)
    parser.add_argument('-k', '--k', dest='k', action='store', type=int, default=10)
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

    k_min, k_max = args.kmin, args.kmax

    if k_min is not None and k_max is not None:
        k_range = range(k_min, k_max + 1)
        plot_x = []
        plot_y = []

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
