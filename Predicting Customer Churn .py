#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: Predicting Customer Churn
# * The data is centred on customer churn, the rate at which a commercial customer will leave the commercial platform
# that they are currently a (paying) customer of a telecommunications company.

# In[76]:


# import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# # Step 1: Reading and Understanding data

# In[77]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[78]:


# Check number of rows and columns
df_train.shape


# In[79]:


df_test.shape


# In[80]:


# Top 5 record
pd.set_option('display.max_columns', 85)
df_train.head()


# In[81]:


df_test.head()


# In[82]:


# Check the summary
df_train.info()


# In[83]:


df_test.info()


# In[84]:


# Statistic of numeric column
df_train.describe()


# In[85]:


df_test.describe()


# # Step 2: Data Cleaning

# ### 1. Checking missing value/treatment of missing value:-

# In[86]:


df_train.isnull().sum()


# In[87]:


df_test.isnull().sum()


# In[88]:


df_train = df_train.drop('state',axis=1)


# In[89]:


df_test = df_test.drop(['state','id'],axis=1)


# ### Observation :-
# * Here no missing value in dataset.

# ### 2.checking outliers/treatment:-

# In[90]:


## checking outliers using boxplot:-

df_train.plot(kind='box', subplots=True, figsize=(22,20), layout=(10,4))
plt.show()


# ##### Next, we do capping to 99 percentile on numeric column of train dataframe :

# In[91]:


# for numarical data
num_df = df_train.loc[:,df_train.dtypes != 'object']


# In[92]:


def num(x):
    plt.figure(figsize=(6,6))
    plt.title(x)
    sns.boxplot(df_train[x])
    plt.show()
    return


# In[93]:


for x in num_df:
    q3,q1 = np.percentile(df_train[x],[75,25])
    q4= np.percentile(df_train[x],[99])
    df_train.loc[df_train[x] > q4[0], x] = q4[0]
    num(x)


# # Step-3. Data analysis 

# In[94]:


sns.countplot(x = df_train.churn)
plt.show()


# ### Multivariate Analysis :

# In[95]:


# Plot heatmap to check the correlation 
sns.set(rc = {'figure.figsize':(20,10)})
sns.heatmap(df_train.corr(),annot=True)


# #### Observation:
# • Good Correction between number_vmail_messages and voice_mail_plan , total_day_charge and total_eve_minutes , total_eve_charge and total_night_minutes , total_night_charge and total_intl_minutes , total_intl_charge and total_intl_minutes.

# # 4.Data Preparation

# ### Encoding

# In[96]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[97]:


cat_col = ['area_code','international_plan','voice_mail_plan','churn']


# In[98]:


for i in cat_col:
    df_train[i]= le.fit_transform(df_train[i])


# In[99]:


df_train.head()


# In[100]:


from sklearn.preprocessing import StandardScaler
scl = StandardScaler()


# ### 2. Check if there is an imbalance in data. If there is an imbalance in data, resolve it.

# In[101]:


from sklearn.utils import resample


# In[102]:


#create two different dataframe of majority and minority class 
df_majority = df_train[(df_train['churn']==0)] 
df_minority = df_train[(df_train['churn']==1)] 


# In[103]:


# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 4250, # to match majority class
                                 random_state=42)  # reproducible results


# In[104]:


# Combine majority class with upsampled minority class
df_train = pd.concat([df_minority_upsampled, df_majority])


# ### Observation :-
# * our target class has an imbalance. 
# * So, we’ll try to upsample the data so that the minority class matches with the majority class.

# ## Train_Test_Split

# In[105]:


## CREATE X and y
X = df_train.drop('churn',axis=1)
Y = df_train['churn']


# In[106]:


#### Here we create TRAIN | VALIDATION | TEST  #########
from sklearn.model_selection import train_test_split


# In[107]:


# 70% of data is training data, set aside other 30%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# ### Scale data

# In[108]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### 3. Build a Logistic Regression classification model which will predict whether a customer is at risk to churn from the platform.

# In[109]:


from sklearn.linear_model import LogisticRegression


# In[110]:


lr_model = LogisticRegression()


# In[111]:


# fit the model
lr_model.fit(X_train,y_train)


# In[112]:


y_pred_lr = lr_model.predict(X_test)
y_pred_lr


# In[113]:


from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


# In[114]:


# Accuracy Score 
metrics.accuracy_score(y_pred_lr,y_test)


# In[115]:


# Confusion Matrix
confusion = confusion_matrix(y_pred_lr,y_test)
confusion


# In[116]:


from sklearn.metrics import classification_report


# In[117]:


print(classification_report(y_test, y_pred_lr))


# In[118]:


TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]


# In[119]:


# Sensitivity
Sensitivity = TP / float(TP+FN)
Sensitivity


# In[120]:


# Specificity
Specificity = TN / float(TN+FP)
Specificity


# ### 4. Build Naive Bayes model which will predict whether a customer is at risk to churn from the platform.

# In[121]:


from sklearn.naive_bayes import GaussianNB


# In[122]:


gnb = GaussianNB(priors=None)


# In[123]:


# fit the model
gnb.fit(X_train,y_train)


# In[124]:


y_pred_gnb = gnb.predict(X_test)
y_pred_gnb


# In[125]:


# Accuracy Score 
metrics.accuracy_score(y_pred_gnb,y_test)


# In[126]:


# Confusion Matrix
confusion = confusion_matrix(y_pred_gnb,y_test)
confusion


# In[127]:


print(classification_report(y_test, y_pred_gnb))


# ### 5. Build a K-nearest classifier which will predict whether a customer is at risk to churn from the platform.

# In[128]:


from sklearn.neighbors import KNeighborsClassifier


# In[129]:


knn_model = KNeighborsClassifier(n_neighbors=1)


# In[130]:


knn_model.fit(X_train,y_train)


# In[131]:


y_pred_knn = knn_model.predict(X_test)


# In[132]:


accuracy_score(y_test,y_pred_knn)


# In[133]:


confusion_matrix(y_test,y_pred_knn)


# In[134]:


print(classification_report(y_test,y_pred_knn))


# ### 6. Find optimal parameters for the algorithm through GridSearchCV and build SVC model which will predict whether a customer is at risk to churn from the platform. 

# In[135]:


from sklearn.svm import SVC


# In[136]:


model = SVC()


# In[137]:


model.fit(X_train, y_train)


# In[138]:


y_pred_svc = model.predict(X_test)
y_pred_svc


# In[139]:


accuracy_score(y_test,y_pred_svc)


# In[140]:


confusion_matrix(y_test,y_pred_svc)


# In[141]:


print(classification_report(y_test,y_pred_svc))


# ###### Find best estimator for model using GridSearchCV

# In[142]:


from sklearn.model_selection import GridSearchCV


# In[143]:


param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}


# In[144]:


grid = GridSearchCV(model,param_grid)


# In[145]:


grid.fit(X_train,y_train)


# In[146]:


grid.best_params_


# In[147]:


model = SVC(C=1 , kernel='rbf')


# In[148]:


model.fit(X_train, y_train)


# In[149]:


y_pred_svc = model.predict(X_test)
y_pred_svc


# In[150]:


accuracy_score(y_test,y_pred_svc)


# In[151]:


confusion_matrix(y_test,y_pred_svc)


# In[152]:


print(classification_report(y_test,y_pred_svc))


# ### 7. Find optimal parameters for the algorithm through GridSearchCV and build a Decision tree which will predict whether a customer is at risk to churn from the platform. 

# In[153]:


from sklearn.tree import DecisionTreeClassifier


# In[154]:


dt = DecisionTreeClassifier()


# In[155]:


# fit the model
dt.fit(X_train,y_train)


# In[156]:


y_pred_dt = dt.predict(X_test)
y_pred_dt


# In[157]:


# Accuracy Score 
metrics.accuracy_score(y_test,y_pred_dt)


# In[158]:


# Confusion Matrix
confusion_matrix(y_test,y_pred_dt)


# In[159]:


print(classification_report(y_test,y_pred_dt))


# ###### Find best estimator for model using GridSearchCV

# In[160]:


from sklearn.model_selection import GridSearchCV


# In[161]:


params = {'max_leaf_nodes': list(range(20, 50)), 'min_samples_split': [20, 30, 40, 50, 60]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=20), params, cv=3)
grid_search_cv.fit(X_train, y_train)


# In[162]:


grid_search_cv.best_estimator_


# In[163]:


df = DecisionTreeClassifier(criterion='entropy',min_samples_split=30,
                           max_leaf_nodes=48,random_state=20)


# In[164]:


dt = dt.fit(X_train,y_train)


# In[165]:


y_pred_dt = dt.predict(X_test)


# In[166]:


metrics.accuracy_score(y_test, y_pred_dt)


# ### 8. Find optimal parameters for the algorithm through RandomSearchCV and build a Random which will predict whether a customer is at risk to churn from the platform. 

# In[167]:


from sklearn.ensemble import RandomForestClassifier


# In[168]:


# Use 10 random trees
model = RandomForestClassifier()


# In[169]:


model.fit(X_train,y_train)


# In[170]:


y_pred_rf = model.predict(X_test)


# In[171]:


confusion_matrix(y_test,y_pred_rf)


# In[172]:


metrics.accuracy_score(y_test, y_pred_rf)


# In[173]:


print(classification_report(y_test,y_pred_rf))


# ###### Find best estimator for model using GridSearchCV

# In[174]:


n_estimators=[64,100,128,200]
max_features= [2,3,4]
bootstrap = [True,False]


# In[175]:


param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             }


# In[176]:


rfc = RandomForestClassifier()
grid = GridSearchCV(rfc,param_grid)


# In[177]:


grid.fit(X_train,y_train)


# In[178]:


grid.best_params_


# In[179]:


model = RandomForestClassifier(bootstrap=False , max_features=2)


# In[180]:


model.fit(X_train,y_train)


# In[181]:


y_pred_rf = model.predict(X_test)
y_pred_rf


# In[182]:


metrics.accuracy_score(y_test, y_pred_rf)


# ### 9. Model Selection: Evaluate and compare performance of all the models to find the best model.

# In[183]:


compare = pd.DataFrame({'Model':['Logistic Regression' ,'Naive Bayes' , 'K-nearest' , 'SVC',
                                'Decision Tree' , 'Random Forest Regression'],
                        'Accuracy':[accuracy_score(y_test,y_pred_lr)*100,accuracy_score(y_test,y_pred_gnb)*100,
                                   accuracy_score(y_test,y_pred_knn)*100,accuracy_score(y_test,y_pred_svc)*100,
                                   accuracy_score(y_test,y_pred_dt)*100,accuracy_score(y_test,y_pred_rf)*100]})


# In[184]:


compare.sort_values(by='Accuracy', ascending=False)

