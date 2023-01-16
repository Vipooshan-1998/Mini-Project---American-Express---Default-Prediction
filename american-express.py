#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#1.-Import-Packages" data-toc-modified-id="1.-Import-Packages-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>1. Import Packages</a></span></li><li><span><a href="#2.-Load-Files" data-toc-modified-id="2.-Load-Files-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>2. Load Files</a></span></li><li><span><a href="#3.-Data-Preparation" data-toc-modified-id="3.-Data-Preparation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>3. Data Preparation</a></span></li><li><span><a href="#5.-Model-Building-and-Evaluation" data-toc-modified-id="5.-Model-Building-and-Evaluation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>5. Model Building and Evaluation</a></span></li><li><span><a href="#6.-Submission-to-Kaggle" data-toc-modified-id="6.-Submission-to-Kaggle-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>6. Submission to Kaggle</a></span></li></ul></div>

# In[1]:


import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Import Packages

# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
import lightgbm
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, accuracy_score
import shap
import itertools


# # 2. Load Files

# In[3]:


train = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/train.parquet").groupby('customer_ID').tail(4)
test = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet").groupby('customer_ID').tail(4)
train_labels = pd.read_csv("../input/amex-default-prediction/train_labels.csv")
train.head()


# In[4]:


#shapes
train.shape, test.shape, train_labels.shape


# In[5]:


#Checking Missing values having more than 40%
columns = train.columns[(train.isna().sum()/len(train))*100>40]


# In[6]:


#Drop the missing columns having more than 40%
train = train.drop(columns, axis=1)
test = test.drop(columns, axis=1)


# In[7]:


#fill the missing values
train = train.bfill(axis='rows').ffill(axis='rows')
test = test.bfill(axis='rows').ffill(axis='rows')


# In[8]:


train.reset_index(inplace=True)
test.reset_index(inplace=True)


# In[9]:


train =train.groupby('customer_ID').tail(1)
test = test.groupby('customer_ID').tail(1)


# In[10]:


train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


# In[11]:


#shape
train.shape, train_labels.shape, test.shape


# In[12]:


#Type Conversion
obj_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
for col in obj_col:
    train[col]=train[col].astype('int').astype('str')
    test[col]=test[col].astype('int').astype('str')
    print(train[col].unique())
    print(test[col].unique())


# # 3. Data Preparation

# In[13]:


train = train.merge(train_labels, how='inner', on="customer_ID")


# In[14]:


train.head()


# In[15]:


test_data = test.copy()
train = train.drop(['index','customer_ID', 'S_2'], axis=1)
test = test.drop(['index','customer_ID', 'S_2'], axis=1)


# In[16]:


#one hot Encoding for categorical features
obj_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
train = pd.get_dummies(train, columns=obj_col, drop_first=True)
test = pd.get_dummies(test, columns=obj_col, drop_first=True)
train.shape, test.shape


# In[17]:


Features=train.loc[:, test.columns]
target = train['target']


# # 5. Model Building and Evaluation

# **Model 1: XGBOOST Classifier**

# In[19]:


XGB = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1).fit(Features, target)


# In[ ]:


np.mean(cross_val_score(XGB, Features, target, scoring='accuracy', cv=5))


# **Accuracy**

# In[ ]:


y_pred = XGB.predict(Features)
accuracy_score(target, y_pred)


# **Confusion Matrix**

# In[ ]:


cm=confusion_matrix(target, y_pred)
plt.figure(figsize=(13,5))
plt.title("Confusion Matrix")
plt.imshow(cm, alpha=0.5, cmap='PuBu')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center")
plt.show()


# **ROC AUC Curve**

# In[ ]:


#AUC (ROC curve)
fpr, tpr, thresholds = roc_curve(target, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=XGB)
display.plot()
plt.show()


# **Classification Report**

# In[ ]:


print(classification_report(target, y_pred))


# **SHAP Summary Plot**

# In[ ]:


explainer = shap.TreeExplainer(XGB)
shap_values = explainer.shap_values(Features)
shap.summary_plot(shap_values, Features)


# SVM and KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[ ]:


KNC = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNC.fit(Features, target)


# In[ ]:


y_pred = KNC.predict(Features)
accuracy_score(target, y_pred)


# In[ ]:


SVCl = SVC(kernel = 'linear', random_state = 0)
SVCl.fit(Features, target)


# In[ ]:


y_pred = SVCl.predict(Features)
accuracy_score(target, y_pred)


# # 6. Submission to Kaggle

# In[ ]:


test_data['prediction']=XGB.predict_proba(test)[:,1]
test_data[['customer_ID','prediction']].to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('pwd')


# In[ ]:




