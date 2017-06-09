import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
import datetime

train = pd.read_csv('train.csv', parse_dates= ['timestamp'])
# parse_dates make the specified column as datetime data type
macro = pd.read_csv('macro.csv', parse_dates= ['timestamp'])
test = pd.read_csv('test.csv', parse_dates= ['timestamp'])

id_test = test.id

#It seems that this doesn't improve anything.

#train["timestamp"] = pd.to_datetime(train["timestamp"])
#train["year"], train["month"], train["day"] = train["timestamp"].dt.year,train["timestamp"].dt.month,train["timestamp"].dt.day

#test["timestamp"] = pd.to_datetime(test["timestamp"])
#test["year"], test["month"], test["day"] = test["timestamp"].dt.year,test["timestamp"].dt.month,test["timestamp"].dt.day

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)
x_test['Source'] = 'Test'
x_train['Source'] = 'Train'

x_all = pd.concat([x_train, x_test],ignore_index=True)

for c in x_all.columns:
    if c != 'Source':
        if x_all[c].dtype == 'object':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))


x_train = x_all[x_all['Source']=='Train']
x_test = x_all[x_all['Source']=='Test']
x_train = x_train.drop('Source', axis=1)
x_test = x_test.drop('Source', axis=1)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'min_child_weight': 5,
    'gamma': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.65,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'reg_alpha': 0.0,
    'seed': 27
}

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_output)
print(num_boost_rounds)
# cv gives the best num_boost_round
# Then tune other params

######## Tune parameters ###############################
## https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# param_grid = {
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }
# model = model_selection.GridSearchCV(estimator=XGBRegressor(learning_rate =0.1, n_estimators=174, max_depth=5,
#                                                             min_child_weight=5, gamma=0, subsample=0.7, colsample_bytree=0.65,
#                                                             objective= 'reg:linear', reg_alpha=0.0, seed=27),
#                                      param_grid=param_grid, n_jobs=-1, verbose=10, cv=3) # n_jobs=-1 uses all cores
#
# model.fit(x_train, y_train)
# model.grid_score, model.best_params_
#######################################################

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
# subsample: 0.5: 2.643640e+06, 0.7: 2.626, 0.9: 6208
#

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv('xgbSub.csv', index=False)