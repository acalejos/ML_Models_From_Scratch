#!/usr/bin/env python
import pandas as pd
import sys
from Models.progressBar import progressBar


def trainModel(tfrac, data):
    train = data.sample(frac=tfrac, random_state=47)
    probs = {}
    headers = list(train)
    for header in headers:
        probs[header] = {}
    for item in train['decision'].unique():
        probs['decision'][item] = train['decision'].value_counts(normalize=True)[
            item]
    headers.remove('decision')
    for header in headers:
        probs[header] = (train.groupby('decision')[
                         header].value_counts().unstack(fill_value=0)).stack()
        probs[header] = (probs[header]+1)/(probs[header].nunique() +
                                           train.groupby('decision')[header].count())
    return (probs, headers)


def makePrediction(probs, event, headers):
    prob0 = probs['decision'][0]
    prob1 = probs['decision'][1]
    for header in headers:
        current = event[header]
        prob0 *= probs[header][0].get(current, 0)
        prob1 *= probs[header][1].get(current, 0)
    if prob0 > prob1:
        return 0
    else:
        return 1


def test(data, probs, headers, name):
    correct = 0
    total = 0
    for index, row in data.iterrows():
        current = row
        actual = current['decision']
        prediction = makePrediction(probs, current, headers)
        if prediction == actual:
            correct += 1
            total += 1
        else:
            total += 1
        if (index % 100 == 0):
            progressBar(name, index, len(data))
    sys.stdout.flush()
    sys.stdout.write("\n")
    accuracy = round(100*float(correct) / float(total), 2)
    print(name + " Accuracy: " + str(accuracy))
    return accuracy


def nbc(tfrac, data, train_fn, test_fn):
    (model, headers) = trainModel(tfrac, data)
    trainBed = pd.read_csv(train_fn)
    testBed = pd.read_csv(test_fn)
    test(trainBed, model, headers, "Training")
    test(testBed, model, headers, "Testing")

# Main function


def main(argv):
    train_fn = argv[0]
    test_fn = argv[1]
    data = pd.read_csv(train_fn)
    tfrac = 1
    nbc(tfrac, data, train_fn, test_fn)


if __name__ == '__main__':
    if len(sys.argv) > 4:
        print("Incorrect arguments")
        # sys.exit(0)
    else:
        main(sys.argv[1:])
