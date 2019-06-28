# CS573 HW2 - Logistic Regression and Support Vector Machines From Scratch
### Author: Andres Alejos

---

## Python Files and How to Call Them:
    - preprocess_assg3.py : python preprocess_assg3.py dating-full.csv
        - Takes in 'dating-full.csv' and a name for the output file and produces 'dating.csv'
    - lr_svm.py : python lr_svm.py [trainingDataFilename] [testDataFilename] [model (1 or 2)]
      - 1 = logistic regression , 2 = SVM
    - cv.py : python cv.py [lr_training_ data] [svm_training_data] [nbc_training_data]
    - 5_1.py : Not called from the command line.  Copy of code used from HW2 called from cv.py
## Notes:
    - I have different testing and training CSV files for each model, which should be used with their specified model
      - They take the form: {lr,nb,svm}_{trainingSet,testSet}.csv
    - nb_trainingSet.csv was computed using label encoding from HW2, using the train/test split from this homework (HW3)
    - I multiplied my accuracies by 100, so if an accuracy is .74 before, my output will print 74. as a percentage
