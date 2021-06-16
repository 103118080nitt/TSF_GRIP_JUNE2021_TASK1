#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: Ram vriksh


# #GRIP @The Sparskfoundation 

# # TASK 1:Prediction using supervised Machine learning

# prediction of precentage of a student based on the number of study hours 
# 
# this is simple linear regression task as it involve only two variables

# In[1]:


#import required libbraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 


# In[2]:


#reading  data 

data=pd.read_csv("student_scores - student_scores.csv")
print("read data successfully")


# In[3]:


data.head(8)


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


# plot the graph using data

data.plot(x='Hours',
          y='Scores',
          style= 'o')
plt.title('Hours Vs score')
plt.xlabel('hours of study')
plt.ylabel('score in percentage')
plt.show()


# checking manually if the data have kind of relation using graph

# from the above graph we can see clearly that the is linear relation between Hours and study

# # Split the data into training and test set 

# In[10]:


X=data.iloc[:,:-1].values
y=data.iloc[:,1].values
 


# In[11]:


#split the data into train test set
X_train, X_test, y_train, y_test = train_test_split(
     X, y, 
     test_size=0.3,
     random_state=100)


# In[12]:


#check the co-relation
data.corr()


# # Train the model

# In[13]:


#import linear model 
#define a regression model and fit the data into it
from sklearn import linear_model
LR_model=linear_model.LinearRegression()

LR_model.fit(X_train,y_train)


# In[14]:



LR_model.intercept_


# In[15]:


LR_model.coef_


# # Predicting the marks using the model

# In[21]:


y_pred=LR_model.predict(X_test)
y_pred


# In[24]:


prediction=pd.DataFrame({'Hours':[i[0] for i in X_test],'Predictions':[k for k in y_test]})
prediction


# In[22]:


y_test


# In[26]:


#comparsion between actual and predicted marks
comparsion=pd.DataFrame({'predicted marks ':y_pred,'Actual marks':y_test})
comparsion


# # Visuallizing the comparsion between predicted and actual marks
# 

# In[27]:


plt.scatter(x=X_test,y=y_test,color='red')
plt.plot(X_test,y_pred,color='green')
plt.title('actual marks v/s predicted marks')
plt.xlabel('study hours')
plt.ylabel('marks got')
plt.show()


# # Evaluate the model 
# 

# In[29]:


from sklearn.metrics import mean_absolute_error
print('mean absolute error in the model ',mean_absolute_error(y_pred,y_test))


# # find out the score if a student study for 9.25hrs/day

# In[36]:


hours=[9.25]
marks_predicted=LR_model.predict([hours])
marks_predicted


# # by this model, we can say that if  students study 9.25hrs/day he/she can get 92.80 precent marks

# In[ ]:




