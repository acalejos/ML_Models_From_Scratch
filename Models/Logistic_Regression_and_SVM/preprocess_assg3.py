#!/usr/bin/env python
import csv
import sys
import pandas as pd
import numpy as np
from numpy import array
from ..utils import progressBar
#np.set_printoptions(threshold=sys.maxsize)
def preprocess(data):
    r = np.arange(6501,len(data))
    data.drop(r)
    headers1 = ['race','race_o','field']
    headers2 = ['field']
    headers3 = ['gender','race','race_o','field']
    headers4 = ['gaming','reading']
    preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
    preference_scores = preference_scores_of_participant + preference_scores_of_partner
    preference_totals = {}
    preference_counts = {}
    encodingLengths = {}
    for score in preference_scores:
        preference_totals[score] = 0
        preference_counts[score] = 0
    encoding_values = {}
    for i,row in data.iterrows():
        for header in headers4:
            current = row[header]
            if int(current) > 10:
                data.ix[i,header] = 10
        for header in headers1:
            current = row[header]
            if current.startswith("'") and current.endswith("'") and len(current) > 1:
                data.ix[i,header] = current.strip("\'")
        for header in headers2:
            current = data.ix[i,header]
            if current[0].isupper():
                current = current.lower()
                data.ix[i,header] = current
        for header in preference_scores:
            current = row[header]
            preference_totals[header] += float(current)
            preference_counts[header] += 1
        for header in preference_scores:
            current = row[header]
            if preference_totals[header] != 0:
                data.ix[i,header] = float(current) / preference_totals[header]
        if (i % 100 == 0):
            progressBar("Preprocessing",i,6500)
        sys.stdout.flush()
    gender_l = sorted(list(data['gender'].unique()))
    (gender_len,gender_index) =  (len(gender_l),gender_l.index('female'))
    gender_vector = getEncoding(gender_len,gender_index)

    race_l = sorted(list(data['race'].unique()))
    (race_len,race_index) =  (len(race_l),race_l.index('Black/African American'))
    race_vector = getEncoding(race_len,race_index)

    raceo_l = sorted(list(data['race_o'].unique()))
    (raceo_len,raceo_index) =  (len(raceo_l),raceo_l.index('Other'))
    raceo_vector = getEncoding(raceo_len,raceo_index)

    field_l = sorted(list(data['field'].unique()))
    (field_len,field_index) =  (len(field_l),field_l.index('economics'))
    field_vector = getEncoding(field_len,field_index)

    for header in headers3:
        dropped = header+"_"+sorted(data[header].unique())[-1]
        data = pd.concat([data,pd.get_dummies(data[header],prefix=header)],axis=1)
        data.drop(header,axis=1,inplace=True)
        data.drop(dropped,axis=1,inplace=True)


    print("\n")
    print("Mapped vector for female in column gender: "+str(gender_vector))
    print("Mapped vector for Black/African American in column race: "+str(race_vector))
    print("Mapped vector for Other in column race_o: "+str(raceo_vector))
    print("Mapped vector for economics in column field: "+str(field_vector))
    return data

def split(data,trainingSet,testSet,fract,write):
    svm_data = data.copy()
    svm_data["decision"] = svm_data["decision"].replace(0,-1)
    svm_test = svm_data.sample(frac=fract,random_state=25)
    svm_train = svm_data.drop(svm_test.index)
    lr_test = data.sample(frac=fract,random_state=25)
    lr_train = data.drop(lr_test.index)
    if write:
        lr_train.to_csv("lr_"+trainingSet,index=False)
        lr_test.to_csv("lr_"+testSet,index=False)
        svm_train.to_csv("svm_"+trainingSet,index=False)
        svm_test.to_csv("svm_"+testSet,index=False)
    #return (train,test)

def getEncoding(length,value):
    l = [x*0 for x in range(0,length-1)]
    if value < len(l):
        l[value] = 1
    return array(l)

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
