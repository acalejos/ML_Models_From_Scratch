#!/usr/bin/env python
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(input_file):
    distinct_labels = {}
    labels = {}
    ratings_labels = ['attractive_partner','sincere_partner','intelligence_partner','funny_partner','ambition_partner','shared_interests_partner']
    for label in ratings_labels:
        labels[label] = [] 
    for row in input_file:
        for header in ratings_labels:
            current = row[header]
            if current not in labels[header]:
                labels[header] = labels[header] + [current] 
    for label in ratings_labels:
        distinct_labels[label] = len(labels[label])
    return (labels,distinct_labels,ratings_labels)

def find_success_rates(input_file, labels, distinct_labels,ratings_labels):
    success_rates = {}
    for key, value in labels.iteritems():
        for label in value:
            success_rates[key] = {label:{'total':0,'count':0,'percentage':0.0}}
    for key, value in labels.iteritems():
        for label in value:
            success_rates[key][label] = {'total':0,'count':0,'percentage':0.0}
            continue
    for row in input_file:
        for header in ratings_labels:
            current = row[header]
            decision = row['decision']
            if int(decision) == 0:
                #No second date
                success_rates[header][current]['count'] += 1
            else:
                #Yes second date
                success_rates[header][current]['total'] += 1
                success_rates[header][current]['count'] += 1
    for key,value in success_rates.items():
        for _, value2 in value.items():
            value2['percentage'] =  float(100*value2['total'])/float(value2['count'])
    #print(success_rates)
    return success_rates


def plot_data(header,current):
    fig,_ = plt.subplots(num=None,figsize=(16,12),dpi=80,facecolor='w',edgecolor='k')
    fig.canvas.set_window_title("Success Rates Based on "+header+" Ratings")
    plt.title("Success Rates Based on "+header+" Ratings")
    new_dict = {}
    for key,value in current.iteritems():
        new_dict[key] = value['percentage']
    items = new_dict.items()
    for i in range(len(items)):
        items[i] = (float(items[i][0]),items[i][1])
    lists = sorted(items)
    x,y = zip(*lists)
    plt.xlabel('Rating from Participant')
    plt.ylabel('Percent of Partners Who Received Second Dates')
    plt.xticks(np.arange(min(x),max(x)+1,1.0))
    plt.yticks(np.arange(min(y),max(y)+1,10.0))
    plt.plot(x,y)
    saveName = "plot/" + header + ".png"
    plt.savefig(saveName)

#Main function
def main(argv):
    f = open(argv[0])
    input_file = csv.DictReader(f)
    labels,distinct_labels,ratings_labels = get_data(input_file)
    f.close()
    f = open(argv[0])
    input_file = csv.DictReader(f)
    success_rates = find_success_rates(input_file,labels,distinct_labels,ratings_labels)
    f.close()
    for header in ratings_labels:
        current = success_rates[header]
        #print(current)
        plot_data(header,current)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
