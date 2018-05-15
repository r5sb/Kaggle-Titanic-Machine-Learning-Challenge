
# coding: utf-8

# In[212]:


#Initial Submission For Kaggle Titanic Machine Learning Challenge

import numpy as np 
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#example_sub = pd.read_csv("../input/gender_submission.csv")


# In[216]:


missing_vals = train.isnull().sum()
missing_vals #Check for features in Test that have too many missing values.


# In[218]:


train_new = train.dropna(subset = ['Embarked'])
# missing_vals = train_new.isnull().sum()
# missing_vals


# In[219]:





# In[220]:


train_new = train_new.fillna(method = 'bfill', axis=0) #Fill Mising Vals with previous value in column.


# In[223]:


tf = train_new.drop(['Cabin','PassengerId','Name','Ticket','Parch','SibSp','Embarked'],axis = 1) 
#print (train_new.shape)
tf.Sex.replace(['female', 'male'], [1, 0], inplace=True)
y_train = tf['Survived']
x_train = tf.drop('Survived',axis=1)
#x_train.head()


# In[224]:


pid_test = test['PassengerId']
tsf = test.drop(['Cabin','PassengerId','Name','Ticket','Parch','SibSp','Embarked'],axis =1)
tsf.Sex.replace(['female', 'male'], [1, 0], inplace=True)
#tsf.head()


# In[225]:


#print (np.mean(tsf['Age']))
tsf = tsf.fillna(method = 'bfill', axis=0).fillna(np.mean(tsf['Age'])) #Replace missing values in test.
tsf.head()


# In[226]:


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


# In[227]:


preds = model.predict(tsf)


# In[228]:


preds.shape


# In[229]:


# print (pid_test.shape)
# print (preds.shape)
final_arr = np.column_stack((pid_test,preds))


# In[230]:


# final_arr.shape


# In[232]:


#Replace output file name
df = pd.DataFrame(final_arr)
df.to_csv("op_mlp_2.csv",header=None,index=None)

