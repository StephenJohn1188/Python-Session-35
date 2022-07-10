#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris=load_iris()


# In[3]:


iris


# In[4]:


iris.feature_names


# In[5]:


iris.keys()


# In[6]:


iris.DESCR


# In[7]:


iris.target


# In[8]:


ds=pd.DataFrame(data=iris.data)
ds
sb.pairplot(ds)


# In[9]:


x=iris.data
y=iris.target
y


# In[ ]:





# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=22,random_state=43)


# In[11]:


mnb=MultinomialNB()
mnb.fit(x_train,y_train)
predmnb=mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# # Decision Tree Classifier

# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc.score(x_train,y_train)
preddtc=dtc.predict(x_test)
print(accuracy_score(y_test,preddtc))
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# # Support Vector Classifier

# In[17]:


from sklearn.svm import SVC


# In[18]:


supp=SVC()
supp.fit(x_train,y_train)
supp.score(x_train,y_train)
predsupp=supp.predict(x_test)
print(accuracy_score(y_test,predsupp))
print(confusion_matrix(y_test,predsupp))
print(classification_report(y_test,predsupp))


# In[23]:


#rbf,poly,linear
supp=SVC(kernel='rbf')
supp.fit(x_train,y_train)
supp.score(x_train,y_train)
predsupp=supp.predict(x_test)
print(accuracy_score(y_test,predsupp))
print(confusion_matrix(y_test,predsupp))
print(classification_report(y_test,predsupp))


# In[24]:


def svmkernel(ker):
    supp=SVC(kernel='rbf')
    supp.fit(x_train,y_train)
    supp.score(x_train,y_train)
    predsupp=supp.predict(x_test)
    print(accuracy_score(y_test,predsupp))
    print(confusion_matrix(y_test,predsupp))
    print(classification_report(y_test,predsupp))


# In[26]:


svmkernel("rbf")


# In[27]:


svmkernel("poly")


# In[ ]:




