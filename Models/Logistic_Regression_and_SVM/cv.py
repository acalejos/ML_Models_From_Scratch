#!/usr/bin/env python
from scipy import stats
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lr_svm
from Models.NaiveBayes import nb


def split(data):
    S = []
    for s in range(0, 3):
        current = data[s].sample(frac=1, random_state=18)
        S1 = []
        ranges = np.linspace(0, len(current), 11)
        for i in range(0, 10):
            S1.append(current[int(ranges[i]):int(ranges[i+1])])
        S.append(S1)
    return S


def crossValidate(S):
    #lr_S = S[0]
    #svm_S = S[1]
    #nb_S = S[2]
    accuracies = {0: {}, 1: {}, 2: {}}  # 1 = lr | 2 = svm | 3 = nbc
    t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    sizes = []
    for s in range(0, 3):
        for t in t_frac:
            accuracies[s][t] = []
    for s in range(0, 3):
        for t in t_frac:
            for idx in range(1, 10):
                # print(idx)
                test_set = S[s][idx]
                SC = list(S[s])
                SC.pop(idx)
                SC = pd.concat(SC, axis=0)
                train_set = SC.sample(frac=t, random_state=32)
                length = len(train_set)
                if length not in sizes:
                    sizes.append(length)
                if s == 0:
                    model = lr_svm.train_lr(train_set)
                    testResult = round(lr_svm.test(
                        test_set, model, 1, .5, "LR Testing"), 2)
                    accuracies[0][t].append(testResult)
                if s == 1:
                    model = lr_svm.train_svm(train_set)
                    testResult = round(lr_svm.test(
                        test_set, model, 2, .5, "SVM Testing"), 2)
                    accuracies[1][t].append(testResult)
                if s == 2:
                    (probs, headers) = nb.trainModel(1.0, train_set)
                    testResult = round(
                        nb.test(test_set, probs, headers, "NBC Testing"), 2)
                    accuracies[2][t].append(testResult)

    return accuracies, sizes


def plot_data(accuracies, sizes):
    t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    '''
    accuracies:
    {0: {0.2: [64.63, 66.98, 67.04, 61.78, 68.89, 65.12, 62.96, 62.89, 65.56], 0.1: [68.52, 57.88, 64.63, 66.23, 65.74, 65.49, 61.85, 61.78, 66.3], 0.025: [67.22, 71.99, 63.52, 69.2, 65.19, 67.16, 61.67, 54.36, 71.3], 0.075: [58.7, 55.66, 63.15, 63.08, 60.93, 61.22, 58.89, 59.0, 63.89], 0.15: [68.7, 60.3, 65.56, 65.49, 66.67, 65.12, 62.96, 62.34, 63.33], 0.05: [66.3, 59.74, 60.0, 60.48, 57.41, 54.36, 54.44, 67.35, 64.26]}, 1: {0.2: [59.26, 60.3, 55.0, 55.66, 56.67, 56.96, 55.37, 56.4, 56.85], 0.1: [59.26, 60.3, 55.0, 55.66, 56.67, 56.96, 55.37, 56.4, 56.85], 0.025: [38.7, 62.15, 53.89, 59.93, 54.44, 58.44, 46.11, 44.9, 44.07], 0.075: [59.26, 60.3, 55.0, 55.66, 56.67, 56.96, 55.37, 56.4, 56.85], 0.15: [59.26, 60.3, 55.0, 55.66, 56.67, 56.96, 55.37, 56.4, 56.85], 0.05: [59.07, 39.89, 54.81, 43.78, 55.93, 42.86, 55.37, 56.4, 56.85]}, 2: {0.2: [0.76, 0.77, 0.78, 0.78, 0.73, 0.75, 0.74, 0.77, 0.74], 0.1: [0.73, 0.75, 0.78, 0.74, 0.72, 0.74, 0.71, 0.75, 0.74], 0.025: [0.62, 0.65, 0.69, 0.62, 0.6, 0.62, 0.58, 0.65, 0.62], 0.075: [0.7, 0.72, 0.75, 0.73, 0.69, 0.71, 0.69, 0.74, 0.74], 0.15: [0.75, 0.77, 0.78, 0.77, 0.73, 0.75, 0.72, 0.76, 0.74], 0.05: [0.66, 0.71, 0.74, 0.69, 0.66, 0.68, 0.66, 0.71, 0.72]}}

    stats:
    {0: {0.2: (65.09444444444443, 0.24168141638605142), 0.1: (64.26888888888888, 0.336797816790592), 0.025: (65.73444444444443, 0.570724303633117), 0.075: (60.50222222222222, 0.28055880833111435), 0.15: (64.49666666666668, 0.26419378098181096), 0.05: (60.48222222222223, 0.4962086096604669)}, 1: {0.2: (56.94111111111111, 0.18485855968924605), 0.1: (56.94111111111111, 0.18485855968924605), 0.025: (51.40333333333333, 0.8595302893116508), 0.075: (56.94111111111111, 0.18485855968924605), 0.15: (56.94111111111111, 0.18485855968924605), 0.05: (51.662222222222226, 0.7629077250370461)}, 2: {0.2: (0.7577777777777778, 0.0019441994750646464), 0.1: (0.74, 0.0020951312035156983), 0.025: (0.6277777777777778, 0.003344744983739276), 0.075: (0.7188888888888889, 0.002368311863364308), 0.15: (0.7522222222222222, 0.0020805308081916956), 0.05: (0.6922222222222222, 0.0030888879020239)}}
    '''

    statistics = {0: {}, 1: {}, 2: {}}
    for i in range(0, 3):
        for t in t_frac:
            current = accuracies[i][t]
            mean = np.mean(current)
            stdev = np.std(current)
            sterr = stdev / len(current)
            statistics[i][t] = (mean, sterr)

    Y = []
    Err = []
    for i in range(0, 3):
        Y.append([])
        Err.append([])
        for _, value in statistics[i].iteritems():
            Y[i].append(round(value[0], 2))
            Err[i].append(round(value[1], 2))

    x = sizes
    fig, _ = plt.subplots(num=None, figsize=(
        16, 12), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title("Cross-Validation Model Accuracy")
    plt.title("Cross-Validation Model Accuracy")
    plt.xlabel('Size of Training Set')
    plt.ylabel('Model Accuracy')
    plt.plot(x, Y[0], c='r', marker="o", fillstyle='none',
             label='Logistic Regression Model Accuracy')
    plt.plot(x, Y[1], c='b', marker="o",
             fillstyle='none', label='SVM Model Accuracy')
    plt.plot(x, Y[2], c='m', marker="o",
             fillstyle='none', label='NBC Model Accuracy')
    plt.errorbar(x, Y[0], yerr=Err[0], c='r', capsize=5)
    plt.errorbar(x, Y[1], yerr=Err[1], c='b', capsize=5)
    plt.errorbar(x, Y[2], yerr=Err[2], c='m', capsize=5)
    plt.legend(loc='upper right')
    saveName = "HW3_Learning_Curves.png"
    plt.savefig(saveName)
    # Conduct Paired T-Test and print results
    results = stats.ttest_rel(Y[0], Y[1])
    print("t-statistic: {0} | p-value: {1}".format(results[0], results[1]))
# Main function


def main(argv):
    lr_filename = argv[0]
    svm_filename = argv[1]
    nb_filename = argv[2]
    lr_data = pd.read_csv(lr_filename)
    svm_data = pd.read_csv(svm_filename)
    nb_data = pd.read_csv(nb_filename)
    data = [lr_data, svm_data, nb_data]
    S = split(data)  # [lr_S, svm_S, nb_S]
    accuracies, sizes = crossValidate(S)
    plot_data(accuracies, sizes)


if __name__ == '__main__':
    if len(sys.argv) > 4:
        print(
            "Usage: python cv.py [lr_training_ data] [svm_training_data] [nbc_training_data]")
        # sys.exit(0)
    else:
        main(sys.argv[1:])
