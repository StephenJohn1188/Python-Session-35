#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


cancerset=load_breast_cancer()
cancerset


# In[3]:


cancerset.keys()


# In[4]:


cancerset.data


# In[5]:


cancerset.target


# In[6]:


cancerset.feature_names


# In[7]:


cancerset.DESCR


# In[8]:


df=pd.DataFrame(data=cancerset.data)


# In[9]:


df


# In[10]:


df=pd.DataFrame(data=cancerset.data, columns=cancerset.feature_names)


# In[11]:


df


# In[12]:


df['target']=pd.DataFrame(cancerset.target)
df


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.head()


# In[16]:


df.tail(10)


# In[17]:


df.sample()


# In[18]:


df.corr()


# In[19]:


sns.heatmap(df.corr(),annot=True)


# In[20]:


sns.countplot(df['target'])


# In[21]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)


# In[22]:


x=df.drop(['target'],axis=1)


# In[23]:


x


# In[24]:


y=df['target']


# In[25]:


y


# In[26]:


x.shape


# In[27]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=20,random_state=45)


# In[28]:


train_x


# In[29]:


test_x


# In[30]:


train_y


# In[31]:


test_y


# In[32]:


lg=LogisticRegression()


# In[33]:


lg.fit(train_x,train_y)


# In[34]:


pred=lg.predict(test_x)


# In[35]:


print(pred)


# In[36]:


print('Accuracy Score=',accuracy_score(test_y,pred))


# In[37]:


print(confusion_matrix(test_y,pred))


# In[38]:


print(classification_report(test_y,pred))


# In[40]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# In[41]:


y_pred_prob=lg.predict_proba(test_x)[:,1]
y_pred_prob


# In[43]:


fpr,tpr,thresholds=roc_curve(test_y,y_pred_prob)


# In[44]:


fpr


# In[45]:


tpr


# In[46]:


thresholds


# In[47]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression')
plt.show()


# In[48]:


auc_score=roc_auc_score(test_y,lg.predict(test_x))


# In[49]:


auc_score


# In[ ]:





# In[ ]:




