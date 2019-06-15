# CS573 HW4 - Decision Trees, Bagging, and Random Forests
### Author: Andres Alejos

---

## Python Files and How to Call Them:
    - preprocess_assg4.py : python preprocess_assg3.py dating-full.csv
        - Takes in 'dating-full.csv' and a outputs "trainingSet.csv" and "testSet.csv"
    - trees.py : python trees.py [trainingDataFilename] [testDataFilename] [model (1,2, or 3)]
      - 1 = Decision Tree , 2 = Bagging, 3 = Random Forests
    - cv_depth.py : python cv.py [training_data]
    - cv_frac.py : python cv.py [training_data]
    - cv_numtrees.py : python cv.py [training_data]
    - progressBar.py: Not called from the command line.  Used across entire homework to show training progress

## Notes:
    - Due to the shallow depth and reduction in features available at each depth, Random Forests sometimes creates models which do not have certain paths
      - In the case that a given entry that we are predicting has features that require a path that does not exist, the model always return a wrong prediction in that case (by design)
      - This causes RF to have lower accuracy
    - Due to the long running time of the cross-validation section, I have provided the accuracies which are outputted by each CV test
      - These accuracies are provided as comments in each cv.py file
      - You can uncomment the hard-coded accuracies to run the plot function
