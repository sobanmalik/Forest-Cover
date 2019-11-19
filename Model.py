#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
from warnings import filterwarnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[2]:


train.shape,test.shape


# In[3]:


train.head()


# In[4]:


train.describe().round(1)


# In[5]:


train.Vertical_Distance_To_Hydrology.plot.hist(alpha=0.7,bins=10,color='blue')


# In[6]:


train.columns[1:11]


# In[7]:


from sklearn.preprocessing import minmax_scale
for i in train.columns[1:11]:
    bins=[-1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    train[i+'_cat']=pd.cut(minmax_scale(train[i]),bins=bins,labels=[j for j in range(10) ])
    train[i+'_cat']=train[i+'_cat'].astype('int64')
    test[i+'_cat']=pd.cut(minmax_scale(test[i]),bins=bins,labels=[j for j in range(10) ])
    test[i+'_cat']=test[i+'_cat'].astype('int64')


# In[8]:


def create_dummies(df,column_name):
    dummy_table=pd.get_dummies(df[column_name],prefix=column_name+'Class')
    df=pd.concat([df,dummy_table],axis=1)
    return df
for i in train.columns[1:11]+'_cat':
    train=create_dummies(train,i)
    test=create_dummies(test,i)


# In[9]:


def waterbody(df):
    df.loc[df['Vertical_Distance_To_Hydrology']<0,'has_waterbody']=1
    df['has_waterbody'].fillna(0,inplace=True)
    df['has_waterbody']=df['has_waterbody'].astype('int64')
waterbody(train)
waterbody(test)
train.head()


# In[10]:


train_final=train.drop('Cover_Type',axis=1)
train.columns[1:65]


# ## Model

# In[34]:


for i in [170]:#,70,90,110,130,170]:
    columns=train_final.columns[11:i]
    train_X,val_X,train_y,val_y=train_test_split(train[columns],train['Cover_Type'],
                                                test_size=0.2,random_state=42)
    
   


# In[ ]:


model=RandomForestClassifier(n_estimators = 719,
                                      max_features = 0.3,
                                      max_depth = 464,
                                      min_samples_split = 2,
                                      min_samples_leaf = 1,
                                      bootstrap = False,
                                      random_state=42)
   model.fit(train_X,train_y)
   print(i,accuracy_score(model.predict(train_X),train_y),accuracy_score(model.predict(val_X),val_y))
   #scores=cross_val_score(model,train_final[columns],train['Cover_Type'],cv=5)
   #accuracy_cv=scores.mean()
   #print(i,accuracy_cv)


# In[18]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
#params training
param_grid = {'n_estimators': np.logspace(2,3.5,8).astype(int),
              'max_features': [0.1,0.3,0.5,0.7,0.9],
              'max_depth': np.logspace(0,3,10).astype(int),
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap':[True, False]}

grid = RandomizedSearchCV(estimator=rf, 
                          param_distributions=param_grid, 
                          n_iter=100, # This was set to 100 in my offline version
                          cv=3, 
                          verbose=3, 
                          n_jobs=1,
                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 
                          refit='NLL')


# In[40]:


#grid.fit(train_final[columns],train['Cover_Type'])
np.logspace(2,3.5,8).astype(int)


# In[ ]:


from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.2,
    #random_strength=0.1,
    depth=8,
    loss_function='Binary',
    eval_metric='f1',
    metric_period = 100,    
    leaf_estimation_method='Newton'
)
model.fit(train_X, train_y,
             eval_set=(val_X,val_y),
             #cat_features=categorical_var,
             use_best_model=True,
             verbose=True)


# In[18]:


a=model.predict(test[columns])


# In[ ]:


stack.predict(val_X.to_numpy)


# In[17]:


accuracy_score(stack.predict(val_X),val_y)


# In[29]:


test['cover_type']=holdout_predictions
test['cover_type'].value_counts()


# In[15]:


get_ipython().run_line_magic('time', '')
from catboost import CatBoostClassifier
cb_model = CatBoostClassifier(iterations=500,
                             # loss_function='Multiclass',
                             learning_rate=0.2,
                             depth=12,
                             eval_metric='Accuracy',
                             random_seed = 23,
                             #bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
cb_model.fit(train_X, train_y,
             eval_set=(val_X,val_y),
             #cat_features=categorical_var,
             use_best_model=True,
             verbose=True)


# In[33]:


accuracy_score(model.predict(train_X),train_y),accuracy_score(model.predict(val_X),val_y)


# In[30]:


scores=cross_val_score(model,train_final[columns],train['Cover_Type'],cv=3)
accuracy_cv=scores.mean()
accuracy_cv


# ## Submission

# In[37]:


holdout_predictions=stack.predict(test[columns].to_numpy())


# In[38]:


test['cover_type']=holdout_predictions.astype('int64')
test['cover_type'].value_counts()


# In[49]:


len((holdout_predictions).astype('int64')),test.shape[0]
submission_dict


# In[39]:


submission_dict={'Id':test['Id'],
              'Cover_Type':test['cover_type']}
submission_df=pd.DataFrame(data=submission_dict)
submission_df.to_csv('Submission.csv',index=False)


# In[36]:


accuracy_score(stack.predict(val_X.to_numpy()),val_y.to_numpy())


# In[35]:


import os
import random
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, 
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

random_state = 1
random.seed(random_state)
np.random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)


print('> Loading data...')
#X_train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')
#X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

y_train = train_y.copy()
X_train = train_X


print('> Setting up classifiers...')
n_jobs = -1  # Use all processor cores to speed things up

ab_clf = AdaBoostClassifier(n_estimators=200,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=3,
                                random_state=random_state),
                            random_state=random_state)

bg_clf = BaggingClassifier(n_estimators=200,
                           random_state=random_state)

gb_clf = GradientBoostingClassifier(n_estimators=400,
                                    min_samples_leaf=3,
                                    tol=0.1,
                                    verbose=0,
                                    random_state=random_state)

lg_clf = LGBMClassifier(n_estimators=400,
                        num_leaves=30,
                        verbosity=0,
                        random_state=random_state,
                        n_jobs=n_jobs)

rf_clf = RandomForestClassifier(n_estimators=800,
                                min_samples_leaf=3,
                                verbose=0,
                                random_state=random_state,
                                n_jobs=n_jobs)

xg_clf = XGBClassifier(n_estimators=400,
                       min_child_weight=3,
                       verbosity=0,
                       random_state=random_state,
                       n_jobs=n_jobs)

ensemble = [('ab', ab_clf),
            ('bg', bg_clf),
            ('gb', gb_clf),
            ('lg', lg_clf),
            ('rf', rf_clf),
            ('xg', xg_clf)]

stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],
                             meta_classifier=rf_clf,
                             cv=5,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=n_jobs)

# TODO: Find best parameters for each classifier

print('> Cross-validating classifiers...')
#test_ids = X_test.index.copy()  # Keep for submitting later
X_train = X_train.to_numpy()  # Converting to numpy matrices because...
#X_test = X_test.to_numpy()  # ...XGBoost complains about dataframe columns

scores = dict()
for label, clf in ensemble:
    print('  -- Cross-validating {} classifier...'.format(label))
    score = cross_val_score(clf, X_train, y_train,
                            cv=5,
                            scoring='accuracy',
                            verbose=1,
                            n_jobs=n_jobs)
    scores[label] = score
    print('  -- {} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
    print()

print('> All cross-validation scores')
for label, score in scores.items():
    print('  -- {} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
print()


print('> Fitting & predicting...')
stack = stack.fit(X_train, y_train)
#prediction = stack.predict(X_test)




print('> Done !')


# In[ ]:


({'n_estimators': 1178,
  'min_samples_split': 5,
  'min_samples_leaf': 1,
  'max_features': 0.5,
  'max_depth': 46,
  'bootstrap': True},
 0.4992911774546519)

