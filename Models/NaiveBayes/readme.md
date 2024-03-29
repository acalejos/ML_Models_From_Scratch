
# CS573 HW2 - Naive-Bayes Classifier From Scratch

### Author: Andres Alejos

---

## Python Files and How to Call Them:

    - preprocess.py : python preprocess.py dating-full.csv dating.csv
        - Takes in 'dating-full.csv' and a name for the output file and produces 'dating.csv'
    - norm_data.py : python norm_data.py dating.csv
        - Takes in 'dating.csv' and produces a bar plot called 'mean_gender_scored.png'
    - visualize.py : python visualize.py dating.csv
        - Takes in 'dating.csv' and produces 6 scatter plots called 'headerName.png'
    - discretize.py : python discretize.py dating.csv dating-binned.csv
        - Takes in 'dating.csv' and a name for the output file  produces 'dating-binned.csv'
    - split.py : python split.py dating-binned.csv trainingSet.csv testSet.csv
        - Takes in 'dating-binned.csv' and outputs two csv files named 'trainingSet.csv' and 'testSet.csv'
    - nb.py : python nb.py [train CSV] [test CSV]
        - Outputs the accuracy of the model using the training and test sets created in split.py
    - nb_cv1.py : python nb_cv1.py dating.csv
         - Takes in 'dating.csv' and outputs the accuracy of the model with various bin sizes
    - nb_cv2.py : python nb_cv2.py  trainingSet.csv
        - Takes in trainingSet.csv and outputs the accuracy of the model when trained on different train/test splits
## Notes:

    - I broke up the NBC into multiple functions for better control:
        - train : trains the model based on the training set and returns the model in the form of a map
        - makePrediction : Takes a single entry to make a prediction for, and takes in the model, and makes a prediction using the Bayes classifier
        - test : Calls 'makePrediction' on all entries in a data set and tracks accuracy of the model
        - nbc : Calls 'trainModel' to train the classifier, then calls 'test' on both the training and test datasets to test accuracy
    - For the testing of the models I added a progrss bar so that the user can tell how long the model has left to work, since it took a while for the model to work depending on the size of the set that is being tested
    - I implemented Laplace Correction with (k=1) inside of the 'trainModel' function in '5_1.py'
