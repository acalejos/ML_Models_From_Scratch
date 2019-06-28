#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from ..utils import progressBar
plt.rcParams.update({'font.size': 18})
np.random.seed(0)


def randomGrayscale(data):
    for i in range(10):
        #Get random row for each digit
        current = data[data[1]==i].sample(1).drop([0,1],axis=1).values.flatten().astype(np.int32).reshape((28,28))
        #Create image from array
        img = Image.fromarray(current)
        img = img.convert('RGB')
        #Save image
        filename = "raw_digits/raw_digit_{}.bmp".format(i)
        img.save(filename)

def randomEmbedded(data):
    samples = []
    for i in np.random.randint(0,len(data),size=1000):
        current = data[data[0] == i].drop(0,axis=1).values.flatten()
        samples.append(current)
    samples = np.array(samples)
    labels = samples[:,0].astype(int)
    points = samples[:,[1,2]]
    x = points[:,0]
    y = points[:,1]
    cdict = {0:'red',1:'blue',2:'green',3:'cyan',4:'magenta',5:'#cc6600',6:'#ff66cc',7:'#4d2600',8:'#cccc00',9:'#66ff66'}
    colors = np.vectorize(cdict.get)(labels)
    figure(figsize=(30,30))
    recs = []
    class_colors = list(cdict.values())
    for i in range(len(class_colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colors[i]))
    plt.scatter(x,y,color=colors)
    plt.title("MNIST Embedding Visualization")
    plt.legend(recs,[str(i) for i in cdict.keys()])
    saveName = "embedded_visualization.png"
    plt.savefig(saveName)

def main():
    rawData = pd.read_csv("digits-raw.csv",header=None)
    embeddedData = pd.read_csv("digits-embedding.csv",header=None)
    randomGrayscale(rawData)
    randomEmbedded(embeddedData)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Incorrect arguments")
        sys.exit()
    else:
        main()
