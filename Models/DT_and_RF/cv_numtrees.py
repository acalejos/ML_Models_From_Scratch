#!/usr/bin/env python
from scipy import stats
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trees
from ..utils import progressBar

def split(data):
    shuffle = data.sample(frac = 1, random_state = 18)
    current = shuffle.sample(frac = 0.5,random_state = 32)
    S = []
    ranges = np.linspace(0,len(current),11)
    for i in range(0,10):
        S.append(current[int(ranges[i]):int(ranges[i+1])])
    return S

def crossValidate(S):
    accuracies = {0:{},1:{}} # 0 = BT | 1 = RF
    num = [10,20,40,50]
    totTime = 2*len(num)*10
    it = 0
    for s in range(0,2):
        for t in num:
            accuracies[s][t] = []
    for s in range(0,2):
        for t in num:
            for idx in range(1,10):
                #print(idx)
                test_set = S[idx]
                SC = list(S)
                SC.pop(idx)
                SC = pd.concat(SC,axis=0)
                if s == 0:
                    models = trees.bootstrap(SC,t,8)
                    testResult = round(trees.testBagging(SC,models,"BT CV_Numtrees"),2)
                    accuracies[0][t].append(testResult)
                if s == 1:
                    models = trees.bootstrapRandom(SC,t,8)
                    testResult = round(trees.testBagging(SC,models,"RF CV_Numtrees"),2)
                    accuracies[1][t].append(testResult)
                it+=1
                sys.stdout.flush()
                progressBar("####### CV_Numtrees Total ########",it,totTime)
    return accuracies

def plot_data(accuracies):
    num = [10,20,40,50]
    statistics = {0:{},1:{}}
    for i in range(0,2):
        for t in num:
            current = accuracies[i][t]
            mean = np.mean(current)
            stdev = np.std(current)
            sterr = stdev / len(current)
            statistics[i][t] = (mean,sterr)

    Y = []
    Err = []
    for i in range(0,2):
        Y.append([])
        Err.append([])
        for key, value in statistics[i].items():
            Y[i].append(round(value[0],2))
            Err[i].append(round(value[1],2))

    x = num
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Cross-Validation Number of Trees")
    plt.title("Cross-Validation Number of Trees")
    plt.xlabel('Number of Trees in Ensemble Model')
    plt.ylabel('Model Accuracy')
    plt.plot(x,Y[0],c='r',marker="o",fillstyle = 'none',label='BT Model Accuracy')
    plt.plot(x,Y[1],c='b',marker="o",fillstyle = 'none',label='RF Model Accurac')
    plt.errorbar(x,Y[0],yerr=Err[0],c='r',capsize=5)
    plt.errorbar(x,Y[1],yerr=Err[1],c='b',capsize=5)
    plt.legend(loc='upper right')
    saveName = "cv_numtrees.png"
    plt.savefig(saveName)
    #Conduct Paired T-Test and print results
    results = stats.f_oneway(accuracies[0][10],accuracies[0][20],accuracies[0][40],accuracies[0][50])
    results1 = stats.f_oneway(accuracies[1][10],accuracies[1][20],accuracies[1][40],accuracies[1][50])

    print("t-statistic: {0} | p-value: {1}".format(results[0],results[1]))
    print("t-statistic: {0} | p-value: {1}".format(results1[0],results1[1]))

#Main function
def main(argv):
    data = pd.read_csv(argv[0])
    S = split(data)
    accuracies = crossValidate(S)
    #accuracies = {0: {10: [76.07, 77.69, 76.5, 76.41, 77.74, 77.44, 76.97, 76.45, 75.94], 20: [78.16, 77.39, 77.86, 76.45, 77.14, 76.67, 77.31, 77.78, 76.67], 40: [77.78, 78.21, 76.58, 78.16, 77.82, 77.39, 77.86, 76.41, 76.88], 50: [77.35, 76.79, 77.05, 76.79, 77.14, 77.09, 78.16, 77.44, 77.09]}, 1: {10: [69.49, 69.83, 69.87, 70.04, 68.55, 69.36, 69.7, 71.2, 72.52], 20: [69.36, 69.49, 69.49, 70.64, 69.62, 71.32, 72.18, 70.09, 69.4], 40: [69.74, 69.53, 69.74, 69.96, 69.19, 68.93, 69.74, 70.13, 69.1], 50: [69.79, 69.36, 69.49, 69.96, 69.19, 68.97, 69.62, 69.87, 68.93]}}
    plot_data(accuracies)
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python cv_numtrees.py [training_ data]")
    else:
        main(sys.argv[1:])
