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
X_train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')
X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

y_train = X_train['Cover_Type'].copy()
X_train = X_train.drop(['Cover_Type'], axis='columns')


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
test_ids = X_test.index.copy()  # Keep for submitting later
X_train = X_train.to_numpy()  # Converting to numpy matrices because...
X_test = X_test.to_numpy()  # ...XGBoost complains about dataframe columns

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
prediction = stack.predict(X_test)


print('> Creating submission...')
submission = pd.DataFrame({'Id': test_ids, 'Cover_Type': prediction})
submission.to_csv('submission.csv', index=False)


print('> Done !')
