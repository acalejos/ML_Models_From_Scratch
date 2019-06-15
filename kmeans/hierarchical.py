#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from progressBar import progressBar
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import cluster
from scipy.cluster.hierarchy import fcluster
from scipy import stats
plt.rcParams.update({'font.size': 18})
np.random.seed(0)
import kmeans
import kmeans_analysis as ka

def subsample(embeddedData):
    frames = []
    for i in range(10):
        #Get random row for each digit
        frames.append(embeddedData[embeddedData[1] == i].sample(10))
    data = pd.concat(frames,ignore_index=True)
    return data.drop(0,axis=1)

def Linkage(X,method):
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Dendrogram With {0} Method".format(method))
    plt.title("Dendrogram With {0} Method".format(method))
    Z = linkage(X,method)
    fig = plt.figure(figsize=(25,10))
    dn = dendrogram(Z)
    saveName = "{}_linkage.png".format(method)
    plt.savefig(saveName)
    return Z

def getClusters(data,f,k):
    clusters = {}
    for i in range(k):
        clusters[i+1] = []
    for i,p in enumerate(f):
        cluster = p
        row = data.iloc[i]
        label,x,y = row[1],row[2],row[3]
        current = [x,y,label]
        clusters[cluster].append(current)
    final = {i:np.array(clusters[i]) for i in clusters}
    return final

def getCentroids(clusters):
    centroids = [[],[]]
    for key,value in clusters.items():
        xmean = np.mean(value[:,0])
        ymean = np.mean(value[:,1])
        centroids[0].append(xmean)
        centroids[1].append(ymean)
    return np.array(centroids)

def part3(data,Z,name):
    wc_stats = {}
    sc_stats = {}
    for k in [2,4,8,16,32]:
        f = fcluster(Z, k, criterion='maxclust')
        clusters = getClusters(data,f,k)
        centroids = getCentroids(clusters)
        no_labels = {i:clusters[i][:,:2] for i in clusters}
        wcssd = kmeans.WC_SSD(no_labels,centroids)
        sc = kmeans.SC(no_labels)
        wc_stats[k] = wcssd
        sc_stats[k] = sc
    print("WC",wc_stats)
    print("SC",sc_stats)
    ka.visualize_step1(wc_stats,"{} WCSSD".format(name))
    ka.visualize_step1(sc_stats, "{} Silhouette Coefficient".format(name))
    return wc_stats,sc_stats



def main(argv):
    #Part 1
    embeddedData = pd.read_csv(argv[0],header=None)
    data = subsample(embeddedData)
    X = data.drop(1,axis=1).values
    single = Linkage(X,'single')
    #Part 2
    complete = Linkage(X,'complete')
    average = Linkage(X,'average')
    #Part 3
    names = ['Single Linkage','Complete Linkage','Average Linkage']
    Z = [single,complete,average]
    for z,name in zip(Z,names):
        wc_stats, sc_stats = part3(data,z,name)
    #Part 4
    #Part 5
    K = [8,8,8]
    for z,name,k in zip(Z,names,K):
        f = fcluster(z, k, criterion='maxclust')
        clusters = getClusters(data,f,k)
        centroids = getCentroids(clusters)
        nmi = kmeans.NMI(clusters,data)
        ka.visualize_step4(clusters,centroids,k,nmi,name)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python hierarchical.py [dataFilename]")
        sys.exit()
    else:
        main(sys.argv[1:])
