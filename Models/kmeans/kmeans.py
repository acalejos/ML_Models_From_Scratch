#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from Models.progressBar import progressBar
from scipy.spatial.distance import pdist, cdist
from scipy import stats
plt.rcParams.update({'font.size': 18})


def kmeans(data, k, seed, visualize=False):
    np.random.seed(seed)
    points = data.iloc[:, [2, 3]].values
    labels = data.iloc[:, 1].values
    maxIter = 50
    N = points.shape[0]  # number of training samples
    numFeatures = points.shape[1]  # x and y coords
    centroids = np.array([]).reshape(numFeatures, 0)
    clusters = {}
    for i in np.random.randint(0, N, size=k):
        centroids = np.c_[centroids, points[i]]

    for step in range(maxIter):
        d = np.array([]).reshape(N, 0)
        for i in range(k):
            # Euclidean Distance
            dist = np.sum((points-centroids[:, i])**2, axis=1)
            d = np.c_[d, dist]
        C = np.argmin(d, axis=1)+1
        temp = {}
        for i in range(k):
            temp[i+1] = np.array([]).reshape(3, 0)
        for i in range(N):
            temp[C[i]] = np.c_[temp[C[i]], np.append(points[i], labels[i])]
        for i in range(k):
            temp[i+1] = temp[i+1].T
        for i in range(k):
            centroids[:, i] = np.mean(temp[i+1][:, :2], axis=0)
        clusters = temp
        progressBar("K-Means Clustering", step, maxIter)
    sys.stdout.flush()

    if visualize:
        color = ['red', 'blue', 'green', 'cyan', 'magenta',
                 '#cc6600', '#ff66cc', '#4d2600', '#cccc00', '#66ff66']
        labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5',
                  'cluster6', 'cluster7', 'cluster8', 'cluster9', 'cluster10']
        for i in range(k):
            plt.scatter(clusters[i+1][:, 0], clusters[i+1][:, 1],
                        c=color[i], label=int(stats.mode(clusters[i+1][:, 2])[0][0]))
        plt.scatter(centroids[0, :], centroids[1, :],
                    s=300, c='yellow', label='Centroids')
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        plt.legend()
        plt.show()
    return clusters, centroids


def WC_SSD(clusters, centroids):
    wcssd = 0
    for key, value in clusters.items():
        centroid = np.array([centroids[0][key-1], centroids[1][key-1]])
        wcssd += np.sum((value-centroid)**2)
    return wcssd
# Silhouette Coefficient -- [-1,1]


def SC(clusters):
    S = []
    for key, value in clusters.items():
        #print("SC Value",value)
        A = pdist(value)
        if A.size == 0:
            A = 0
        A = np.unique(A)
        A = np.mean(A)
        others = {i: clusters[i] for i in clusters if i != key}
        B = [cdist(value, others[other]) for other in others]
        B = [np.mean(i) for i in B]
        B = np.mean(B)
        s = (B-A) / np.maximum(A, B)
        S.append(s)
        progressBar("SC", key, len(clusters))
    sys.stdout.flush()
    sc = np.mean(S)
    return sc

# Normalized Mutual Information Gain -- [0,1]
# P(c,g) = number of examples of class c in cluster g / total number of examples in the dataset
# p(c) is the number of examples with this class c / total number of examples in the dataset
# p(g) is the number of examples in this cluster g / total number of examples in the dataset


def NMI(clusters, data):
    classLabels = data[1].unique()
    IG = 0
    i = 1
    for i, c in enumerate(classLabels):
        for _, value in clusters.items():
            labelled = pd.DataFrame(
                {'x': value[:, 0], 'y': value[:, 1], 'label': value[:, 2]})
            total = len(data)
            vc = labelled['label'].value_counts().get(float(c), -1)
            pcg = (vc / total) if vc != -1 else 0
            pc = data[1].value_counts()[c] / total
            pg = len(labelled) / total
            if pcg == 0:
                IG += 0
            else:
                IG += (pcg * np.log(pcg/(pc*pg)))

        progressBar("NMI", i, len(classLabels))

    HC = 0
    for c in classLabels:
        pc = data[1].value_counts()[c] / len(data)
        HC += pc * np.log(pc)
    HC = -HC
    HG = 0
    for _, value in clusters.items():
        labelled = pd.DataFrame(
            {'x': value[:, 0], 'y': value[:, 1], 'label': value[:, 2]})
        pg = len(labelled) / len(data)
        HG += pg * np.log(pg)
    HG = - HG

    nmi = IG / (HC + HG)
    return nmi


def evaluate(clusters, centroids, data):
    no_labels = {i: clusters[i][:, :2] for i in clusters}
    wcssd = WC_SSD(no_labels, centroids)
    sc = SC(no_labels)
    nmi = NMI(clusters, data)
    print("\n")
    print("WC-SSD: {}".format(wcssd))
    print("SC: {}".format(sc))
    print("NMI: {}".format(nmi))


def main(argv):
    embeddedData = pd.read_csv(argv[0], header=None)
    k = int(argv[1])
    clusters, centroids = kmeans(embeddedData, k, 0, False)
    print(centroids)
    evaluate(clusters, centroids, embeddedData)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Usage: python kmeans.py [dataFilename] [# of clusters]")
        sys.exit()
    else:
        main(sys.argv[1:])
