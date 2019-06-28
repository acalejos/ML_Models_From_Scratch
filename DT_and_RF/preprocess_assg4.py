#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from numpy import array
from utils.progressBar import progressBar
################################################
def preprocess(data):
    r = np.arange(6501,len(data))
    data.drop(r,inplace=True)
    headers1 = ['race','race_o','field']
    headers3 = ['gender']
    headers4 = ['gaming','reading']
    preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
    preference_scores = preference_scores_of_participant + preference_scores_of_partner
    preference_totals = {}
    for header in preference_scores:
        preference_totals[header] = 0
    for header in headers1:
        data.drop(header,axis=1,inplace=True)
    for header in preference_scores:
        preference_totals[header] = data[header].sum()
    for i,row in data.iterrows():
        for header in headers4:
            current = row[header]
            if int(current) > 10:
                data.at[i,header] = 10
        for header in headers3:
            data.at[i,header] = 0 if row[header] == 'female' else 1
        for header in preference_scores:
            current = row[header]
            if preference_totals[header] != 0:
                data.at[i,header] = current / preference_totals[header]
        if (i % 100 == 0):
            progressBar("Preprocessing",i,6500)
    sys.stdout.flush()
    #Label Encoding
    labels = [0,1]
    discrete_labels = ['gender','samerace','decision']
    cont_labels = (list(data))
    for label in discrete_labels:
        cont_labels.remove(label)
    for label in cont_labels:
        #print(label)
        data[label] = pd.cut(data[label],bins=2,labels=labels,include_lowest = True)
    return data

def split(data,trainingSet,testSet,fract,write):
    test = data.sample(frac=fract,random_state=47)
    train = data.drop(test.index)
    if write:
        train.to_csv(trainingSet,index=False)
        test.to_csv(testSet,index=False)

def main(argv):
    data = pd.read_csv(argv[0])
    newData = preprocess(data)
    split(newData,"trainingSet.csv","testSet.csv",0.2,True)

if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Incorrect arguments")
        sys.exit(0)
    else:
        main(sys.argv[1:])
