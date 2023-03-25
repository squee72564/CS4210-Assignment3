#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the testing data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

highest_accuracy = 0.0
best_params = {}

#4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c:
    for degree_value in degree:
        for kernel_value in kernel:
            for dfs_value in decision_function_shape:
                
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=dfs_value)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                total_correct = 0
                for x_testSample, y_testSample in zip(X_test, y_test):
                    if clf.predict([x_testSample]) == y_testSample:
                        total_correct += 1

                #calculate accuracy
                accuracy = float(total_correct) / float(len(y_test))

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together with the SVM hyperparameters
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_params = {'C': c_value, 'degree': degree_value, 'kernel': kernel_value, 'decision_function_shape': dfs_value}
                    print(f"Highest SVM accuracy so far: {highest_accuracy:.5f}, Parameters: C={c_value}, degree={degree_value}, kernel={kernel_value}, decision_function_shape={dfs_value}")

print(f"\nBest SVM accuracy: {highest_accuracy:.5f}, Parameters: C={best_params['C']}, degree={best_params['degree']}, kernel={best_params['kernel']}, decision_function_shape={best_params['decision_function_shape']}")
