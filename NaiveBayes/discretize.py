#!/usr/bin/env python
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def discretize(data, out_filename, numBins, write=True):
    discrete_labels = ['gender', 'race',
                       'race_o', 'samerace', 'field', 'decision']
    cont_labels = (list(data))
    for label in discrete_labels:
        cont_labels.remove(label)
    #print(cont_labels)
    range_1 = np.linspace(18, 58, numBins+1)
    range_2 = np.linspace(0, 10, numBins+1)
    range_3 = np.linspace(0, 1, numBins+1)
    range_4 = np.linspace(-1, 1, numBins+1)
    ranges = {'age': range_1, 'age_o': range_1, 'importance_same_race': range_2, 'importance_same_religion': range_2, 'pref_o_attractive': range_3, 'pref_o_sincere': range_3, 'pref_o_intelligence': range_3, 'pref_o_funny': range_3, 'pref_o_ambitious': range_3, 'pref_o_shared_interests': range_3, 'attractive_important': range_3, 'sincere_important': range_3, 'intelligence_important': range_3, 'funny_important': range_3, 'ambition_important': range_3, 'shared_interests_important': range_3, 'attractive': range_2, 'sincere': range_2, 'intelligence': range_2, 'funny': range_2, 'ambition': range_2,
              'attractive_partner': range_2, 'sincere_partner': range_2, 'intelligence_partner': range_2, 'funny_partner': range_2, 'ambition_partner': range_2, 'shared_interests_partner': range_2, 'sports': range_2, 'tvsports': range_2, 'exercise': range_2, 'dining': range_2, 'museums': range_2, 'art': range_2, 'hiking': range_2, 'gaming': range_2, 'clubbing': range_2, 'reading': range_2, 'tv': range_2, 'theater': range_2, 'movies': range_2, 'concerts': range_2, 'music': range_2, 'shopping': range_2, 'yoga': range_2, 'interests_correlate': range_4, 'expected_happy_with_sd_people': range_2, 'like': range_2}
    labels = []
    for i in range(0, numBins):
        labels.append(str(i))
    for label in cont_labels:
        data[label] = pd.cut(data[label], ranges[label],
                             labels=labels, include_lowest=True)
        if write:
            counts = data[label].value_counts().tolist()
            print(label + ": [" + str(counts[0]) + " " + str(counts[1]) + " " +
                  str(counts[2]) + " " + str(counts[3]) + " " + str(counts[4]) + "]")
    if write:
        data.to_csv(out_filename, index=False)
    return data

#Main function


def main(argv):
    data = pd.read_csv(argv[0])
    discretize(data, argv[1], 5)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
