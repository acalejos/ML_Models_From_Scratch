#!/usr/bin/env python
import sys
import pandas as pd

def split(data,trainingSet,testSet,fract,write):
    test = data.sample(frac=fract,random_state=47)
    train = data.drop(test.index)
    if write:
        filepath = "../CSV/Produced/"
        train.to_csv(filepath+trainingSet,index=False)
        test.to_csv(filepath+testSet,index=False)
    return (train,test)

#Main function
def main(argv):
    data = pd.read_csv(argv[0])
    split(data,argv[1],argv[2],0.2,True)

if __name__ == '__main__':
    if len(sys.argv) > 4:
        print("Incorrect arguments")
        #sys.exit(0)
    else:
        main(sys.argv[1:])
