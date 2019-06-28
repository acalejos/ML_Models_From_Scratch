#!/usr/bin/env python
import sys
import random
import pandas as pd
import numpy as np
from numpy import array
from Models.progressBar import progressBar


############## Decision Trees #############
def decisionTree(trainingData, testData):
    attributes = set(list(trainingData.drop("decision", axis=1)))
    root = buildTree(trainingData, attributes, 1, 8)
    testTree(trainingData, root, "DT Training")
    testTree(testData, root, "DT Test")


def makePrediction(model, row, header=""):
    if not model:
        return 2
    if 'decision' in model.keys():
        return model['decision']
    elif (0 in model.keys() or 1 in model.keys()):
        key = row[header]
        return makePrediction(model[key], row, header)
    else:
        key = next(iter(model.keys()))
        return makePrediction(model[key], row, key)


def testTree(data, model, name):
    total = 0
    correct = 0
    i = 0
    for index, row in data.iterrows():
        current = row
        actual = current['decision']
        prediction = makePrediction(model, current)
        if prediction == actual:
            correct += 1
            total += 1
        else:
            total += 1
        if (i % 100 == 0):
            progressBar(name, i, len(data))
        i += 1
    sys.stdout.flush()
    sys.stdout.write("\n")
    accuracy = round(100*float(correct) / float(total), 2)
    print(name + " Accuracy: " + str(accuracy))
    return accuracy


def giniIndex(S):
    counts = S['decision'].value_counts(normalize=True)
    sum = 0
    for i in range(len(counts)):
        sum += counts[counts.index[i]]**2
    return 1 - sum


def giniGain(S, attribute):
    counts = S[attribute].value_counts()
    total = len(S)
    sum = 0
    for i in range(len(counts)):
        sum += (counts[counts.index[i]]/total) * \
            giniIndex(S.loc[S[attribute] == counts.index[i]])
    return giniIndex(S) - sum


def bestAttribute(examples, attributes):
    bestAtt = random.sample(attributes, 1)[0]
    bestGini = 0
    for attribute in attributes:
        gg = giniGain(examples, attribute)
        if gg > bestGini:
            bestAtt = attribute
            bestGini = gg
    newAttributes = attributes - {bestAtt}
    return (bestAtt, newAttributes)


def buildTree(examples, attributes, depth, depthLimit):
    if len(examples) == 0:
        return
    if examples['decision'].nunique() == 1:
        # return leaf node with label
        label = examples['decision'].unique()[0]
        return {'decision': label}
    if not attributes or (depth >= depthLimit) or (len(examples) <= 51):
        # return leaf node with majority label in examples
        label = examples['decision'].value_counts(
            ascending=True).index.tolist()[-1]
        return {'decision': label}
    else:
        attribute, newAttributes = bestAttribute(
            examples, attributes)  # attribute A has 2 possible values
        # Create an internal node with 2 children
        node = {attribute: {0: buildTree(examples.loc[examples[attribute] == 0], newAttributes, depth+1, depthLimit), 1: buildTree(
            examples.loc[examples[attribute] == 1], newAttributes, depth+1, depthLimit)}}
        return node
############# Bagging #####################


def bagging(trainingData, testData):
    m = 30
    models = bootstrap(trainingData, m, 8)
    testBagging(trainingData, models, "BT Training")
    testBagging(testData, models, "BT Test")


def bootstrap(trainingData, m, depthLimit):
    models = []
    for i in range(m):
        current = trainingData.sample(frac=1, replace=True)
        attributes = set(list(current.drop("decision", axis=1)))
        models.append(buildTree(current, attributes, 1, depthLimit))
        progressBar("Bootstrap DT", i, m)
    return models


def makePredictions(models, row, header=""):
    # print(models)
    num0 = 0
    num1 = 0
    for model in models:
        prediction = makePrediction(model, row)
        if prediction == 0:
            num0 += 1
        else:
            num1 += 1
    return 0 if num0 > num1 else 1


def testBagging(data, models, name):
    total = 0
    correct = 0
    i = 0
    for index, row in data.iterrows():
        current = row
        actual = current['decision']
        prediction = makePredictions(models, current)
        if prediction == actual:
            correct += 1
            total += 1
        else:
            total += 1
        if (i % 100 == 0):
            progressBar(name, i, len(data))
        i += 1
    sys.stdout.flush()
    sys.stdout.write("\n")
    accuracy = round(100*float(correct) / float(total), 2)
    print(name + " Accuracy: " + str(accuracy))
    return accuracy

######### Random Forests ################


def randomForests(trainingData, testData):
    m = 30
    models = bootstrapRandom(trainingData, m, 8)
    # print(models)
    testBagging(trainingData, models, "RF Training")
    testBagging(testData, models, "RF Test")


def bootstrapRandom(trainingData, m, depthLimit):
    models = []
    for i in range(m):
        current = trainingData.sample(frac=1, replace=True)
        attributes = set(list(current.drop("decision", axis=1)))
        models.append(buildRandomTree(current, attributes, 1, depthLimit))
        progressBar("Bootstrap RF", i, m)
    return models


def buildRandomTree(examples, attributes, depth, depthLimit):
    if len(examples) == 0:
        return
    if examples['decision'].nunique() == 1:
        # return leaf node with label
        label = examples['decision'].unique()[0]
        return {'decision': label}
    if not attributes or (depth >= depthLimit) or (len(examples) <= 51):
        # return leaf node with majority label in examples
        label = examples['decision'].value_counts(
            ascending=True).index.tolist()[-1]
        return {'decision': label}
    else:
        p = len(attributes)
        downsample = int(round(np.sqrt(p)))
        attributes = set(random.sample(attributes, downsample))
        attribute, newAttributes = bestAttribute(
            examples, attributes)  # attribute A has 2 possible values
        # Create an internal node with 2 children
        node = {attribute: {0: buildTree(examples.loc[examples[attribute] == 0], newAttributes, depth+1, depthLimit), 1: buildTree(
            examples.loc[examples[attribute] == 1], newAttributes, depth+1, depthLimit)}}
        return node


def main(argv):
    trainingData = pd.read_csv(argv[0])
    testData = pd.read_csv(argv[1])
    modelIDX = int(argv[2])
    if modelIDX == 1:
        decisionTree(trainingData, testData)
    elif modelIDX == 2:
        bagging(trainingData, testData)
    else:
        randomForests(trainingData, testData)


if __name__ == '__main__':
    if len(sys.argv) > 4:
        print(
            "Usage: python trees.py [trainingDataFilename] [testDataFilename] [modelIDX]")
        print("       modelIDX 1: Decision Tree")
        print("       modelIDX 2: Bagging")
        print("       modelIDX 3: Random Forest")
    else:
        main(sys.argv[1:])
