#!/usr/bin/env python
import sys
import pandas as pd
import discretize
import split
import numpy as np
import matplotlib.pyplot as plt

nb = __import__('5_1')

def testEffects(in_filename,out_filename):
    bins = [2,5,10,50,100,200]
    trainResults = []
    testResults = []
    for b in bins:
        temp = pd.read_csv(in_filename)
        current = discretize.discretize(temp,out_filename,b,False)
        (trainSet,testSet) = split.split(current,"trainingSet.csv","testSet.csv",0.2,False)
        (probs,headers) = nb.trainModel(1,trainSet)
        print("Bin size: " + str(b))
        trainResult = nb.test(trainSet,probs,headers,"Training Data Set")
        trainResults.append(trainResult)
        testResult = nb.test(testSet,probs,headers,"Test Data Set")
        testResults.append(testResult)
    return (bins,trainResults,testResults)

def plot_data(bins,trainResults,testResults):
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Model Accuray by Bin Size")
    plt.title("Model Accuracy by Bin Size")
    x = bins
    y1 = trainResults
    y2 = testResults
    plt.xlabel('Size of Bin')
    plt.ylabel('Model Accuracy')
    plt.plot(x,y1,c='r',marker="s",label='Model Accuracy on Training Set')
    plt.plot(x,y2,c='b',marker="o",label='Model Accuracy on Test Set')
    plt.legend(loc = 'lower right')
    saveName = "5_2.png"
    plt.savefig(saveName)

#Main function
def main(argv):
    #data = pd.read_csv(argv[0])
    out_filename = "dating-binned.csv"
    (bins,trainResults,testResults) = testEffects(argv[0],out_filename)
    plot_data(bins,trainResults,testResults)

if __name__ == '__main__':
    if len(sys.argv) > 4:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
