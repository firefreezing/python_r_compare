#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Fei: what the difference between Pipeline and make_pipline? 
from sklearn.pipeline import Pipeline
# from lightgbm import LGBMClassifier
# from category_encoders import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from warnings import filterwarnings
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
filterwarnings('ignore')
import os

#%%
print(os.getcwd())
#%%
dat_train = pd.read_csv("./data/clinvar_conflicting_train.csv")
dat_test = pd.read_csv("./data/clinvar_conflicting_test.csv")

print("train shape", dat_train.shape)
print("test shape", dat_test.shape)

dat_train.info()
#%%
x_train = dat_train.drop('class', 1)
y_train = dat_train['class']#.astype(object)

x_test = dat_test.drop('class', 1)
y_test = dat_test['class']#.astype(object)

#%%
categorical_features = ['clnvc', 'consequence', 'impact']

categorical_transformer = Pipeline(
    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))]
)

numeric_features = ['af_esp', 'af_exac', 'af_tgp', 'cadd_phred', 'cadd_raw', 
                    'chrom', 'loftool', 'pos', 'strand']

numeric_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'median'))#,
        #('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ]
)


#%%
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.1, 'min_samples_leaf': 10, 'random_state': 3}

clf = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('gbm_classifier', ensemble.GradientBoostingClassifier(**params))
        #('classifier', LogisticRegression(solver='lbfgs'))
    ]
)


#%%
clf.fit(x_train, y_train)

#%%
print("model score: %.3f" % clf.score(x_test, y_test))