#!/usr/bin/env python
# coding: utf-8

# In[151]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('data.csv')
#df.dtypes

y=df['sqft']
for i in range(y.shape[0]):
    if y[i] > 1:
        y[i]=1

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
data=df.values[:,0]#price
length=data.size
for i in range(length):
    if data[i]<100:
        data[i]=0
    elif data[i]<200:
        data[i]=1
    elif data[i]<300:
        data[i]=2
    elif data[i]<400:
        data[i]=3
    elif data[i]<500:
        data[i]=4
    else:
        data[i]=5
#print(data)
data_=data[:, np.newaxis]
#print(data_)
y=data_
data1=df.values[:,1:18]
data1_=data1[:, np.newaxis]
#print(data1_)
X=data1_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train=y_train.ravel()
y_test=y_test.ravel()
gnb=GaussianNB()
nsamples, nx, ny=X_train.shape
d2_X_train=X_train.reshape((nsamples,nx*ny))
model=gnb.fit(d2_X_train, y_train)
nsamples_, nx_, ny_=X_test.shape
d2_X_test=X_test.reshape((nsamples_,nx_*ny_))
#print(d2_X_test)
#print(y_test)
y_pred=gnb.predict(d2_X_test)
accuracy=gnb.score(d2_X_test,y_test)
#print(y_pred)
print("accuracy (GaussianNB):", accuracy)
'''
y_pred1=gnb.predict_proba(d2_X_test)
accuracy=gnb.score(d2_X_test,y_test)
print(y_pred1)
print("accuracy:", accuracy)
y_pred2=gnb.predict_log_proba(d2_X_test)
accuracy=gnb.score(d2_X_test,y_test)
print(y_pred2)
print("accuracy:", accuracy)
'''

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
model=clf.fit(d2_X_train, y_train)
y_pred=clf.predict(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
#print(y_pred)
print("accuracy (MultinomialNB):", accuracy)
'''
y_pred1=clf.predict_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred1)
print("accuracy:", accuracy)
y_pred2=clf.predict_log_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred2)
print("accuracy:", accuracy)
'''

#Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()
model=clf.fit(d2_X_train, y_train)
y_pred=clf.predict(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
#print(y_pred)
print("accuracy (BernoulliNB):", accuracy)
'''
y_pred1=clf.predict_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred1)
print("accuracy:", accuracy)
y_pred2=clf.predict_log_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred2)
print("accuracy:", accuracy)
'''

#Complement Naive Bayes
from sklearn.naive_bayes import ComplementNB
clf=ComplementNB()
model=clf.fit(d2_X_train, y_train)
y_pred=clf.predict(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
#print(y_pred)
print("accuracy (ComplementNB):", accuracy)
'''
y_pred1=clf.predict_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred1)
print("accuracy:", accuracy)
y_pred2=clf.predict_log_proba(d2_X_test)
accuracy=clf.score(d2_X_test,y_test)
print(y_pred2)
print("accuracy:", accuracy)
'''


# In[154]:


test_data=[[0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0, 2/18, 2.0, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0, 2/18, 2.0, 1.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
clf1=GaussianNB()
model1=clf1.fit(d2_X_train, y_train)
fin=clf1.predict(test_data)
print (clf1.predict(test_data))
num=0
for num in range(14):
    if num == 0:
        print ("empty house: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 1:
        print ("only with internet: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 2:
        print ("only with air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 3:
        print ("only with television: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 4:
        print ("only with Satellite/Cable: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 5:
        print ("only with heater: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 6:
        print ("only with washing machine: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 7:
        print ("only with elevator: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 8:
        print ("only with kitchen: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 9:
        print ("only with hot tub: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 10:
        print ("only with fitness room: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 11:
        print ("only with concierge: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 12:
        print ("only with parking: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")
    if num == 13:
        print ("internet & air-conditioning: ")
        if fin[num] ==0:
            print ("< 100")
        elif fin[num] ==1:
            print ("100 ~ 200")
        elif fin[num] ==2:
            print("200 ~ 300")
        elif fin[num]==3:
            print("300 ~ 400")
        elif fin[num] ==4:
            print("400 ~ 500")
        elif fin[num]==5:
            print("500 ~")


# In[ ]:




