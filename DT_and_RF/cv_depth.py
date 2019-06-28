#!/usr/bin/env python
from scipy import stats
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trees
from utils.progressBar import progressBar

def split(data):
    shuffle = data.sample(frac = 1, random_state = 18)
    current = shuffle.sample(frac = 0.5,random_state = 32)
    S = []
    ranges = np.linspace(0,len(current),11)
    for i in range(0,10):
        S.append(current[int(ranges[i]):int(ranges[i+1])])
    return S

def crossValidate(S):
    accuracies = {0:{},1:{},2:{}} # 0 = DT | 1 = BT | 2 = RF
    d = [3,5,7,9]
    totTime = 3*len(d)*10
    it = 0
    for s in range(0,3):
        for depth in d:
            accuracies[s][depth] = []
    for s in range(0,3):
        for depth in d:
            for idx in range(1,10):
                #print(idx)
                test_set = S[idx]
                SC = list(S)
                SC.pop(idx)
                SC = pd.concat(SC,axis=0)
                if s == 0:
                    attributes = set(list(SC.drop("decision",axis = 1)))
                    model = trees.buildTree(SC,attributes,1,depth)
                    testResult = round(trees.testTree(SC,model,"DT CV_Depth"),2)
                    accuracies[0][depth].append(testResult)
                if s == 1:
                    m = 30
                    models = trees.bootstrap(SC,m,depth)
                    testResult = round(trees.testBagging(SC,models,"BT CV_Depth"),2)
                    accuracies[1][depth].append(testResult)
                if s == 2:
                    m = 30
                    models = trees.bootstrapRandom(SC,m,depth)
                    testResult = round(trees.testBagging(SC,models,"RF CV_Depth"),2)
                    accuracies[2][depth].append(testResult)
                it+=1
                sys.stdout.flush()
                progressBar("####### CV_Depth Total ########",it,totTime)
    return accuracies

def plot_data(accuracies):
    d = [3,5,7,9]

    statistics = {0:{},1:{},2:{}}
    for i in range(0,3):
        for depth in d:
            current = accuracies[i][depth]
            mean = np.mean(current)
            stdev = np.std(current)
            sterr = stdev / len(current)
            statistics[i][depth] = (mean,sterr)

    Y = []
    Err = []
    for i in range(0,3):
        Y.append([])
        Err.append([])
        for _, value in statistics[i].items():
            Y[i].append(round(value[0],2))
            Err[i].append(round(value[1],2))

    x = d
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Cross-Validation Depth")
    plt.title("Cross-Validation Depth")
    plt.xlabel('Depth of Trees')
    plt.ylabel('Model Accuracy')
    plt.plot(x,Y[0],c='r',marker="o",fillstyle = 'none',label='DT Model Accuracy')
    plt.plot(x,Y[1],c='b',marker="o",fillstyle = 'none',label='BT Model Accuracy')
    plt.plot(x,Y[2],c='m',marker="o",fillstyle = 'none',label='RF Model Accuracy')
    plt.errorbar(x,Y[0],yerr=Err[0],c='r',capsize=5)
    plt.errorbar(x,Y[1],yerr=Err[1],c='b',capsize=5)
    plt.errorbar(x,Y[2],yerr=Err[2],c='m',capsize=5)
    plt.legend(loc='upper right')
    saveName = "cv_depth.png"
    plt.savefig(saveName)
    #Conduct Paired T-Test and print results
    results = stats.f_oneway(accuracies[0][3],accuracies[0][5],accuracies[0][7],accuracies[0][9])
    results1 = stats.f_oneway(accuracies[1][3],accuracies[1][5],accuracies[1][7],accuracies[1][9])
    results2 = stats.f_oneway(accuracies[2][3],accuracies[2][5],accuracies[2][7],accuracies[2][9])

    print("t-statistic: {0} | p-value: {1}".format(results[0],results[1]))
    print("t-statistic: {0} | p-value: {1}".format(results1[0],results1[1]))
    print("t-statistic: {0} | p-value: {1}".format(results2[0],results2[1]))
#Main function
def main(argv):
    data = pd.read_csv(argv[0])
    S = split(data)
    accuracies = crossValidate(S)
    #accuracies = {0: {3: [72.95, 72.74, 73.08, 73.33, 72.26, 72.61, 73.25, 73.5, 72.65], 5: [73.8, 73.63, 74.02, 73.85, 72.78, 73.93, 74.1, 74.4, 73.08], 7: [75.34, 74.7, 75.13, 75.0, 74.87, 75.3, 75.51, 75.47, 74.36], 9: [76.5, 77.05, 76.62, 76.62, 76.11, 76.5, 77.18, 77.09, 76.28]}, 1: {3: [72.95, 72.74, 73.08, 73.33, 72.26, 72.61, 73.25, 73.5, 72.65], 5: [73.8, 73.33, 74.23, 74.32, 72.61, 73.63, 74.15, 73.68, 73.03], 7: [75.9, 75.34, 75.64, 76.2, 76.37, 76.03, 75.9, 76.28, 75.13], 9: [78.42, 78.38, 78.59, 79.23, 78.72, 78.85, 78.68, 78.55, 78.59]}, 2: {3: [69.27, 69.15, 69.4, 69.83, 68.5, 68.8, 69.44, 69.62, 68.89], 5: [69.27, 69.23, 69.44, 69.91, 68.5, 68.97, 69.44, 69.79, 68.89], 7: [69.57, 69.91, 69.7, 70.26, 68.68, 69.06, 70.04, 69.79, 69.1], 9: [69.96, 69.36, 70.04, 70.56, 69.4, 69.32, 70.26, 69.83, 69.49]}}
    plot_data(accuracies)
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python cv_depth.py [training_ data]")
    else:
        main(sys.argv[1:])
