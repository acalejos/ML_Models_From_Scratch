#!/usr/bin/env python
import sys
import pandas as pd
import discretize
import split
import numpy as np
import matplotlib.pyplot as plt
nb = __import__('5_1')

def testEffects(in_filename,out_filename):
    fracs = [0.01,0.1,0.2,0.5,0.6,0.75,0.9,1]
    trainResults = []
    testResults = []
    for f in fracs:
        temp = pd.read_csv(in_filename)
        (t,trainSet) = split.split(temp,"trainingSet.csv","testSet.csv",f,False)
        testSet = pd.read_csv("testSet.csv")
        (probs,headers) = nb.trainModel(1.0,trainSet)
        print("Fraction size: " + str(f))
        trainResult = nb.test(trainSet,probs,headers,"Training Data Set")
        trainResults.append(trainResult)
        testResult = nb.test(testSet,probs,headers,"Test Data Set")
        testResults.append(testResult)
    return (fracs,trainResults,testResults)

def plot_data(bins,trainResults,testResults):
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Model Accuray by Training Fraction Size")
    plt.title("Model Accuracy by Training  Fraction Size")
    x = bins
    y1 = trainResults
    y2 = testResults
    plt.xlabel('Fraction Used in Train/Test Split')
    plt.ylabel('Model Accuracy')
    plt.plot(x,y1,c='r',marker="s",label='Model Accuracy on Training Set')
    plt.plot(x,y2,c='b',marker="o",label='Model Accuracy on Test Set')
    plt.legend(loc='lower right')
    saveName = "5_3.png"
    plt.savefig(saveName)

#Main function
def main(argv):
    out_filename = "dating-binned.csv"
    (bins,trainResults,testResults) = testEffects(argv[0],out_filename)
    plot_data(bins,trainResults,testResults)
if __name__ == '__main__':
    if len(sys.argv) > 4:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
