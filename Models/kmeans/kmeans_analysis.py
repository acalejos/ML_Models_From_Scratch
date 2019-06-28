#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from ..utils import progressBar
from scipy.spatial.distance import pdist, cdist
from scipy import stats
plt.rcParams.update({'font.size': 18})
np.random.seed(0)
import kmeans
from math import isnan

def visualize_step1(stats,name):
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("{} With Respect to Size of K".format(name))
    plt.title("{} With Respect to Size of K".format(name))
    plt.xlabel('Size of K')
    plt.ylabel(name)
    x = list(stats)
    y = list(stats.values())
    plt.plot(x,y)
    plt.legend([name],loc='upper left')
    saveName = "{}.png".format(name.replace(" ","_"))
    plt.savefig(saveName)

def step1(dataset,seed,name,visualize):
    wc_stats = {}
    sc_stats = {}
    K = [2,4,8,16,32]
    for i,k in enumerate(K):
        clusters,centroids = kmeans.kmeans(dataset,k,seed,False)
        no_labels = {i:clusters[i][:,:2] for i in clusters}
        wcssd = kmeans.WC_SSD(no_labels,centroids)
        sc = kmeans.SC(no_labels)
        wc_stats[k] = wcssd
        sc_stats[k] = sc
        progressBar("Analysis Step 1",i,len(K))
    if visualize:
        visualize_step1(wc_stats,"({}) WCSSD".format(name))
        visualize_step1(sc_stats, "({}) Silhouette Coefficient".format(name))
    return wc_stats,sc_stats

def visualize_step2(metric,name):
    clean_metrics = []
    for m in metric:
        clean_dict = {k: m[k] for k in m if not isnan(m[k])}
        clean_metrics.append(clean_dict)
    metric_stats = {}
    for key in list(clean_metrics[0]):
        metric2 = [w[key] for w in clean_metrics]
        metric_stats[key] = (np.mean(metric2),np.std(metric2))
    Y = []
    Err = []
    for key,value in metric_stats.items():
        Y.append(round(value[0],2))
        Err.append(round(value[1],2))
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title(name)
    plt.title(name)
    plt.xlabel('Size of K')
    plt.ylabel(name)
    x = metric_stats.keys()
    y = Y
    plt.plot(x,y,c='r',marker="o",fillstyle = 'none',label=name)
    plt.errorbar(x,y,yerr=Err,c='r',capsize=5)
    plt.legend([name],loc='upper left')
    saveName = "{}.png".format(name.replace(" ","_"))
    plt.savefig(saveName)

def visualize_step4(clusters,centroids,k,nmi,name):
    color = ['red','blue','green','cyan','magenta','#cc6600','#ff66cc','#4d2600','#cccc00','#66ff66']
    color = color[:k]
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("{0} Cluster with k = {1} --> NMI = {2}".format(name,k,nmi))
    for i in range(k):
        plt.scatter(clusters[i+1][:,0],clusters[i+1][:,1],c=color[i],label=int(stats.mode(clusters[i+1][:,2])[0][0]))
    plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow',label='Centroids')
    plt.title("{0} Cluster with k = {1} --> NMI = {2}".format(name,k,nmi))
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.legend()
    saveName = "{}_NMI.png".format(name.replace(" ","_"))
    plt.savefig(saveName)

def step4(dataset,seed,name,k):
    clusters,centroids = kmeans.kmeans(dataset,k,seed,False)
    #print(clusters)
    nmi = kmeans.NMI(clusters,dataset)
    clusters,centroids = kmeans.kmeans(dataset.sample(1000),k,seed,False)
    visualize_step4(clusters,centroids,k,nmi,name)

def main(argv):
    embeddedData = pd.read_csv(argv[0],header=None)
    dataset1 = embeddedData
    dataset2 = embeddedData.loc[embeddedData[1].isin([2,4,6,7])]
    dataset3 = embeddedData.loc[embeddedData[1].isin([6,7])]
    datasets = [dataset1,dataset2,dataset3]
    names = ['Dataset 1','Dataset 2','Dataset 3']
    #Step 1
    for dataset,name in zip(datasets,names):
        step1(dataset,0,name,True)
    #Step 2 (Chosen k-values) K = [2,4,8,16,32]
    # Dataset 1 : k = 8 <-- Elbow of wc curve
    # Dataset 2 : k = 4 <-- Elbow of wc curve
    # Dataset 3 : k = 2 <-- Elbow of wc curve
    #Step 3
    for dataset,name in zip(datasets,names):
        wc = []
        sc = []
        for i in np.random.randint(100, size=10):
            wc_stats, sc_stats = step1(dataset,i,name,False)
            wc.append(wc_stats)
            sc.append(sc_stats)
            #progressBar("Analysis Step 2",i,10)
        visualize_step2(wc,"({}) Average WCSSD With Respect to K".format(name))
        visualize_step2(sc,"({}) Average SC With Respect to K".format(name))
    #Step 4
    k_vals = [8,4,2]
    for dataset,name,k in zip(datasets,names,k_vals):
        step4(dataset,0,name,k)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python kmeans_analysis.py [dataFilename]")
        sys.exit()
    else:
        main(sys.argv[1:])
