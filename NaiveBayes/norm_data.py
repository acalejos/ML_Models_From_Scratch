#!/usr/bin/env python
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_means(input_file):
    mean_male = {}
    mean_female = {}
    totals_male = {}
    totals_female = {}
    counts_male = {}
    counts_female = {}
    preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important'] 
    for score in preference_scores_of_participant:
        totals_male[score] = 0
        totals_female[score] = 0
        counts_male[score] = 0
        counts_female[score] = 0
    for row in input_file:
        for header in preference_scores_of_participant:
            current = row[header]
            if int(row['gender']) == 1:
                totals_male[header] += float(current)
                counts_male[header] += 1
            else:
                totals_female[header] += float(current)
                counts_female[header] += 1
    for score in preference_scores_of_participant:
        if counts_male[score] != 0:
            mean_male[score] = totals_male[score] / counts_male[score]
        if counts_female[score] != 0:
            mean_female[score] = totals_female[score] / counts_female[score]
    return (mean_male,mean_female)

def plot_means(mean_male,mean_female):
    X = np.arange(len(mean_male))
    fig,ax = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Mean Scores of Preference by Gender")
    ax.bar(X,mean_male.values(),width=0.2,color='SkyBlue',align='center')
    ax.bar(X-0.2,mean_female.values(),width=0.2,color='IndianRed',align='center')
    ax.legend(('Males','Females'))
    plt.xticks(X,mean_male.keys())
    plt.title("Mean Scores of Preference by Gender")
    plt.savefig("2_1.png")

#Main function
def main(argv):
    f = open(argv[0])
    input_file = csv.DictReader(f)
    (mean_male,mean_female) = get_means(input_file)
    plot_means(mean_male,mean_female)
    f.close() 

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
