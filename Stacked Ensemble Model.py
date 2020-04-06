import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor)
from sklearn import tree
from sklearn import linear_model
from sklearn.cross_validation import KFold

train=pd.read_csv('C:/Users/tuteja urvashi/Desktop/Test.csv')
test=train
x_train= train.iloc[:,1:4].values
y_train=train.iloc[:,4:6].values
x_test=x_train
y_test =y_train

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, reg, seed=0, params=None):
        params['random_state'] = seed
        self.reg = reg(**params)

    def train(self, x_train, y_train):
        self.reg.fit(x_train, y_train)

    def predict(self, x):
        return self.reg.predict(x)

    def fit(self, x, y):
        return self.reg.fit(x, y)

    def feature_importances(self, x, y):
        print(self.reg.fit(x, y).feature_importances_)

def get_oof(reg, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,2))
    oof_test = np.zeros((ntest,2))
    oof_test_skf = np.empty((NFOLDS, ntest,2))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index,:]
        y_tr = y_train[train_index,:]
        x_te = x_train[test_index,:]

        reg.fit(x_tr, y_tr)

        oof_train[test_index,:] = reg.predict(x_te)
        oof_test_skf[i, :] = reg.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 2), oof_test.reshape(-1, 2)

for i in enumerate(kf):
    print(i)


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

#Linear regression Parameters
ln_params= {
    'fit_intercept': True
}

#Decision Tree Parameters
dt_params= {
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt'
}

#Defining regressors
rf = SklearnHelper(reg=RandomForestRegressor, seed=SEED, params=rf_params)
et = SklearnHelper(reg=ExtraTreesRegressor, seed=SEED, params=et_params)
dt=  SklearnHelper(reg=ExtraTreesRegressor, seed=SEED, params=dt_params)
ln = linear_model.LinearRegression()

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
dt_oof_train, dt_oof_test = get_oof(dt, x_train, y_train, x_test) # Decision Trees
ln_oof_train, ln_oof_test= get_oof(ln, x_train,y_train,x_test)    # Linear Regression

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)


#Second Level Prediction from base models
x_train = np.concatenate(( et_oof_train, rf_oof_train, dt_oof_train, ln_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, dt_oof_test,ln_oof_test), axis=1)

rf_f = RandomForestRegressor(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4).fit(x_train, y_train)
predictions = rf_f.predict(x_test)

