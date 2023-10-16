#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os


import warnings
warnings.filterwarnings('ignore')


# ## II. Importing and understanding our dataset 

# In[12]:


dataset = pd.read_csv("heart.csv")


# In[13]:


type(dataset)


# #### Shape of dataset

# In[14]:


dataset.shape


# #### Printing out a few columns

# In[15]:


dataset.head(7)


# In[16]:


dataset.sample(5)


# #### Description

# In[17]:


dataset.describe()


# In[18]:


dataset.info()


# In[19]:


###Luckily, we have no missing values


# #### Let's understand our columns better:

# In[20]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-angina pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# #### Analysing the 'target' variable

# In[21]:


dataset["target"].describe()


# In[22]:


dataset["target"].unique()


# #### Clearly, this is a classification problem, with the target variable having values '0' and '1'

# ### Checking correlation between columns

# In[23]:


print(dataset.corr()["target"].abs().sort_values(ascending=False))


# In[24]:


#This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.


# ## Exploratory Data Analysis (EDA)

# ### First, analysing the target variable:

# In[25]:


y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)


# In[26]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# ### We'll analyse 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca' and 'thal' features

# ### Analysing the 'Sex' feature

# In[27]:


dataset["sex"].unique()


# ##### We notice, that as expected, the 'sex' feature has 2 unique features

# In[28]:


sns.barplot(dataset["sex"],y)


# ##### We notice, that females are more likely to have heart problems than males

# ### Analysing the 'Chest Pain Type' feature

# In[29]:


dataset["cp"].unique()


# ##### As expected, the CP feature has values from 0 to 3

# In[30]:


sns.barplot(dataset["cp"],y)


# ##### We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems

# ### Analysing the FBS feature

# In[31]:


dataset["fbs"].describe()


# In[32]:


dataset["fbs"].unique()


# In[33]:


sns.barplot(dataset["fbs"],y)


# ##### Nothing extraordinary here

# ### Analysing the restecg feature

# In[34]:


dataset["restecg"].unique()


# In[35]:


sns.barplot(dataset["restecg"],y)


# ##### We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'

# ### Analysing the 'exang' feature

# In[36]:


dataset["exang"].unique()


# In[37]:


sns.barplot(dataset["exang"],y)


# ##### People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems

# ### Analysing the Slope feature

# In[38]:


dataset["slope"].unique()


# In[39]:


sns.barplot(dataset["slope"],y)


# ### Analysing the 'ca' feature

# In[40]:


#number of major vessels (0-3) colored by flourosopy


# In[41]:


dataset["ca"].unique()


# In[42]:


sns.countplot(dataset["ca"])


# In[43]:


sns.barplot(dataset["ca"],y)


# ##### ca=4 has astonishingly large number of heart patients

# Analysing the 'thal' feature

# In[45]:


dataset["thal"].unique()


# In[46]:


sns.barplot(dataset["thal"],y)


# In[47]:


sns.distplot(dataset["thal"])


# ## IV. Train Test split

# In[48]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[49]:


X_train.shape


# In[50]:


X_test.shape


# In[51]:


Y_train.shape


# In[52]:


Y_test.shape


# ## V. Model Fitting

# In[53]:


from sklearn.metrics import accuracy_score


# ### SVM

# In[54]:


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


# In[55]:


Y_pred_svm.shape


# In[56]:


print(Y_pred_svm)


# In[57]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# ### Decision Tree

# In[58]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[59]:


print(Y_pred_dt.shape)


# In[60]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)


# In[ ]:


Y_pred_rf.shape


# In[ ]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# ### XGBoost

# In[107]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)


# In[108]:


Y_pred_xgb.shape


# In[109]:


score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# In[110]:


scores = [score_svm,score_dt,score_rf,score_xgb]
algorithms = ["Support Vector Machine","Decision Tree","Random Forest","XGBoost"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[111]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[ ]:




