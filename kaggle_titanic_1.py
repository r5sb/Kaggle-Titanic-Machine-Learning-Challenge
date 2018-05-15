
#Initial Submission For Kaggle Titanic Machine Learning Challenge
#Sidharth Makhija-2018

import numpy as np 
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#example_sub = pd.read_csv("../input/gender_submission.csv")

missing_vals = train.isnull().sum()
missing_vals #Check for features in Test that have too many missing values.

train_new = train.fillna(method = 'bfill', axis=0) #Fill Mising Vals with previous value in column.

tf = train_new.drop(['Cabin','PassengerId','Name','Ticket','Parch','SibSp','Embarked'],axis = 1) 
#print (train_new.shape)
tf.Sex.replace(['female', 'male'], [1, 0], inplace=True)
y_train = tf['Survived']
x_train = tf.drop('Survived',axis=1)
#x_train.head()

pid_test = test['PassengerId']
tsf = test.drop(['Cabin','PassengerId','Name','Ticket','Parch','SibSp','Embarked'],axis =1)
tsf.Sex.replace(['female', 'male'], [1, 0], inplace=True)
#tsf.head()

#print (np.mean(tsf['Age']))
tsf = tsf.fillna(method = 'bfill', axis=0).fillna(np.mean(tsf['Age'])) #Replace missing values in test.
tsf.head()


# Choose your model by commenting out lines

#Comment out for KNN
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train, y_train) 

# SVC(kernel="linear", C=0.025),
# SVC(gamma=2, C=1),
# MLPClassifier(alpha=1)

#Comment out for SVM
# model = SVC(gamma=2,C=1)
# model.fit(x_train, y_train)

#Comment out for MLP
model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 2), random_state=1)
model.fit(x_train, y_train)

preds = model.predict(tsf)

#preds.shape
# print (pid_test.shape)
# print (preds.shape)

final_arr = np.column_stack((pid_test,preds))
# final_arr.shape
#Replace output file name
df = pd.DataFrame(final_arr)
df.to_csv("op_mlp_2.csv",header=None,index=None)

