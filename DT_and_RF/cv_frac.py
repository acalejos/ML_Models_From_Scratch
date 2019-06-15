#!/usr/bin/env python
from scipy import stats
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trees
from progressBar import progressBar

def split(data):
    current = data.sample(frac = 1, random_state = 18)
    S = []
    ranges = np.linspace(0,len(current),11)
    for i in range(0,10):
        S.append(current[int(ranges[i]):int(ranges[i+1])])
    return S

def crossValidate(S):
    accuracies = {0:{},1:{},2:{}} # 0 = DT | 1 = BT | 2 = RF
    t_frac = [0.05,0.075,0.1,0.15,0.2]
    totTime = 3*len(t_frac)*10
    it = 0
    for s in range(0,3):
        for frac in t_frac:
            accuracies[s][frac] = []
    for s in range(0,3):
        for frac in t_frac:
            for idx in range(1,10):
                #print(idx)
                test_set = S[idx]
                SC = list(S)
                SC.pop(idx)
                SC = pd.concat(SC,axis=0)
                train_set = SC.sample(frac = frac, random_state = 32)
                if s == 0:
                    attributes = set(list(train_set.drop("decision",axis = 1)))
                    model = trees.buildTree(train_set,attributes,1,8)
                    testResult = round(trees.testTree(train_set,model,"DT CV_Frac"),2)
                    accuracies[0][frac].append(testResult)
                if s == 1:
                    m = 30
                    models = trees.bootstrap(train_set,m,8)
                    testResult = round(trees.testBagging(train_set,models,"BT CV_Frac"),2)
                    accuracies[1][frac].append(testResult)
                if s == 2:
                    m = 30
                    models = trees.bootstrapRandom(train_set,m,8)
                    testResult = round(trees.testBagging(train_set,models,"RF CV_Frac"),2)
                    accuracies[2][frac].append(testResult)
                it+=1
                sys.stdout.flush()
                progressBar("####### CV_Frac Total ########",it,totTime)
    return accuracies

def plot_data(accuracies):
    t_frac = [0.05,0.075,0.1,0.15,0.2]
    statistics = {0:{},1:{},2:{}}
    for i in range(0,3):
        for t in t_frac:
            current = accuracies[i][t]
            mean = np.mean(current)
            stdev = np.std(current)
            sterr = stdev / len(current)
            statistics[i][t] = (mean,sterr)

    Y = []
    Err = []
    for i in range(0,3):
        Y.append([])
        Err.append([])
        for key, value in statistics[i].items():
            Y[i].append(round(value[0],2))
            Err[i].append(round(value[1],2))

    x = t_frac
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Cross-Validation Fraction")
    plt.title("Cross-Validation Fraction")
    plt.xlabel('Fraction Trained')
    plt.ylabel('Model Accuracy')
    plt.plot(x,Y[0],c='r',marker="o",fillstyle = 'none',label='DT Model Accuracy')
    plt.plot(x,Y[1],c='b',marker="o",fillstyle = 'none',label='BT Model Accuracy')
    plt.plot(x,Y[2],c='m',marker="o",fillstyle = 'none',label='RF Model Accuracy')
    plt.errorbar(x,Y[0],yerr=Err[0],c='r',capsize=5)
    plt.errorbar(x,Y[1],yerr=Err[1],c='b',capsize=5)
    plt.errorbar(x,Y[2],yerr=Err[2],c='m',capsize=5)
    plt.legend(loc='upper right')
    saveName = "cv_frac.png"
    plt.savefig(saveName)
    #Conduct Paired T-Test and print results
    results = stats.ttest_rel(Y[0],Y[1])
    print("t-statistic: {0} | p-value: {1}".format(results[0],results[1]))
#Main function
def main(argv):
    data = pd.read_csv(argv[0])
    S = split(data)
    accuracies = crossValidate(S)
    #accuracies = {0: {0.05: [73.93, 76.5, 73.5, 75.21, 74.36, 76.92, 73.08, 78.21, 77.78], 0.075: [76.64, 77.21, 76.07, 74.36, 77.78, 77.21, 77.21, 77.78, 78.06], 0.1: [77.99, 76.07, 77.35, 78.63, 79.49, 79.27, 77.99, 78.63, 77.78], 0.15: [75.36, 78.06, 77.35, 79.2, 77.64, 77.21, 76.92, 77.21, 78.35], 0.2: [74.68, 75.53, 75.85, 75.21, 75.96, 75.43, 75.96, 76.6, 76.82]}, 1: {0.05: [82.48, 81.2, 78.63, 78.63, 80.77, 81.62, 79.49, 79.91, 82.48], 0.075: [81.2, 80.91, 78.35, 81.48, 80.34, 79.77, 78.92, 80.91, 80.34], 0.1: [78.63, 82.05, 79.27, 80.98, 79.91, 80.34, 78.85, 79.91, 78.42], 0.15: [80.77, 79.63, 81.48, 81.05, 80.63, 78.92, 80.06, 78.49, 79.91], 0.2: [78.1, 79.06, 77.24, 77.03, 78.53, 78.95, 78.42, 78.85, 76.82]}, 2: {0.05: [71.37, 74.79, 71.79, 73.08, 73.5, 72.22, 76.92, 74.36, 75.64], 0.075: [71.79, 71.79, 74.07, 74.64, 73.22, 71.51, 75.21, 76.64, 75.21], 0.1: [73.5, 70.51, 74.79, 73.29, 70.73, 72.86, 71.79, 72.86, 75.0], 0.15: [70.51, 70.37, 70.94, 71.94, 70.23, 70.51, 71.37, 71.37, 73.5], 0.2: [69.12, 69.34, 68.16, 68.38, 69.23, 69.66, 69.34, 71.05, 72.01]}}
    plot_data(accuracies)
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python cv_frac.py [training_ data]")
    else:
        main(sys.argv[1:])
