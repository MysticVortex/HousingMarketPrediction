
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv', parse_dates= ['timestamp'])
# parse_dates make the specified column as datetime data type
macro = pd.read_csv('macro.csv', parse_dates= ['timestamp'])
test = pd.read_csv('test.csv', parse_dates= ['timestamp'])

clean = False
if clean:
    ############### Cleaning errors and noise
    # CORRECTIONS RULES FOR FULL_SQ AND LIFE_SQ (APPLY TO TRAIN AND TEST):
    # IF LIFE SQ >= FULL SQ MAKE FULL SQ NP.NAN
    # IF LIFE SQ < 5 NP.NAN
    # IF FULL SQ < 5 NP.NAN
    # KITCH SQ < LIFE SQ
    # IF KITCH SQ == 0 OR 1 NP.NAN
    # CHECK FOR OUTLIERS IN LIFE SQ, FULL SQ AND KITCH SQ
    # LIFE SQ / FULL SQ MUST BE CONSISTENCY (0.3 IS A CONSERVATIVE RATIO)

    bad_index = train[train.life_sq > train.full_sq].index
    train.ix[bad_index, "life_sq"] = np.NaN

    equal_index = [601,1896,2791]
    test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]

    bad_index = test[test.life_sq > test.full_sq].index
    test.ix[bad_index, "life_sq"] = np.NaN

    bad_index = train[train.life_sq < 5].index
    train.ix[bad_index, "life_sq"] = np.NaN

    bad_index = test[test.life_sq < 5].index
    test.ix[bad_index, "life_sq"] = np.NaN

    bad_index = train[train.full_sq < 5].index
    train.ix[bad_index, "full_sq"] = np.NaN

    bad_index = test[test.full_sq < 5].index
    test.ix[bad_index, "full_sq"] = np.NaN

    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]

    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.ix[bad_index, "kitch_sq"] = np.NaN

    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.ix[bad_index, "kitch_sq"] = np.NaN

    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.ix[bad_index, "kitch_sq"] = np.NaN

    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.ix[bad_index, "kitch_sq"] = np.NaN

    bad_index = train[(train.full_sq > 210) * (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN

    bad_index = test[(test.full_sq > 150) * (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "full_sq"] = np.NaN

    bad_index = train[train.life_sq > 300].index
    train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

    bad_index = test[test.life_sq > 200].index
    test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

    bad_index = train[train.build_year < 1500].index
    train.ix[bad_index, "build_year"] = np.NaN

    bad_index = test[test.build_year < 1500].index
    test.ix[bad_index, "build_year"] = np.NaN

    # CHECK NUM OF ROOMS
    # IS THERE A OUTLIER ?
    # A VERY SMALL OR LARGE NUMBER ?
    # LIFE SQ / ROOM > MIN ROOM SQ (LET'S SAY 5 SQ FOR A ROOM MIGHT BE OK)
    # IF NUM ROOM == 0 SET TO NP.NAN
    # DETECT ABNORMAL NUM ROOMS GIVEN LIFE AND FULL SQ

    bad_index = train[train.num_room == 0].index
    train.ix[bad_index, "num_room"] = np.NaN

    bad_index = test[test.num_room == 0].index
    test.ix[bad_index, "num_room"] = np.NaN

    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.ix[bad_index, "num_room"] = np.NaN

    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN

    # CHECK FLOOR AND MAX FLOOR
    # FLOOR == 0 AND MAX FLOOR == 0 POSSIBLE ??? WE DON'T HAVE IT IN TEST SO NP.NAN
    # FLOOR == 0 0R MAX FLOOR == 0 ??? WE DON'T HAVE IT IN TEST SO NP.NAN (NP.NAN IF MAX FLOOR == 0 FOR BOTH TEST TRAIN)
    # CHECK FLOOR < MAX FLOOR (IF FLOOR > MAX FLOOR -> MAX FLOOR NP.NAN)
    # CHECK FOR OUTLIERS

    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.ix[bad_index, ["max_floor", "floor"]] = np.NaN

    bad_index = train[train.floor == 0].index
    train.ix[bad_index, "floor"] = np.NaN

    bad_index = train[train.max_floor == 0].index
    train.ix[bad_index, "max_floor"] = np.NaN

    bad_index = test[test.max_floor == 0].index
    test.ix[bad_index, "max_floor"] = np.NaN

    bad_index = train[train.floor > train.max_floor].index
    train.ix[bad_index, "max_floor"] = np.NaN

    bad_index = test[test.floor > test.max_floor].index
    test.ix[bad_index, "max_floor"] = np.NaN

    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN

    bad_index = train[train.state == 33].index
    train.ix[bad_index, "state"] = np.NaN


sns.distplot(train['price_doc'])

# deal with label skewness
train['LogAmt']=np.log(train.price_doc+1.0)
sns.distplot(train['LogAmt'])
# There are two pikes in this plot, dig this out later

## Merge data into one dataset to prepare compare between train and test
train_1 = train.copy()
train_1['Source']='Train'
test_1 = test.copy()
test_1['Source']='Test'
alldata = pd.concat([train_1, test_1],ignore_index=True)

macro.columns = ['mac__'+c if c!='timestamp' else 'timestamp' for c in macro.columns ]
alldata=alldata.merge(macro,on='timestamp',how='left')
print(alldata.shape)

# # The function compare train/test data and check if some variable is illy behaved. It is modified a little to fit this dataset to compared between normal/fraud subset.
# It can be applied to both numeric and object data types:
# When the data type is object, it will output the value count of each categories
# When the data type is numeric, it will output the quantiles
# It also seeks any missing values in the dataset
def var_desc(dt,alldata):
    print('--------------------------------------------')
    for c in alldata.columns:
        if alldata[c].dtype==dt:
            t1 = alldata[alldata.Source=='Train'][c]
            t2 = alldata[alldata.Source=='Test'][c]
            if dt=="object":
                f1 = t1[pd.isnull(t1)==False].value_counts()
                f2 = t2[pd.isnull(t2)==False].value_counts()
            else:
                f1 = t1[pd.isnull(t1)==False].describe()
                f2 = t2[pd.isnull(t2)==False].describe()
            m1 = t1.isnull().value_counts()
            m2 = t2.isnull().value_counts()
            f = pd.concat([f1, f2], axis=1)
            m = pd.concat([m1, m2], axis=1)
            f.columns=['Train','Test']
            m.columns=['Train','Test']
            print(dt+' - '+c)
            print('UniqValue - ',len(t1.value_counts()),len(t2.value_counts()))
            print(f.sort_values(by='Train',ascending=False))
            print()

            m_print=m[m.index==True]
            if len(m_print)>0:
                print('missing - '+c)
                print(m_print)
            else:
                print('NO Missing values - '+c)
            if dt!="object":
                if len(t1.value_counts())<=10:
                    c1 = t1.value_counts()
                    c2 = t2.value_counts()
                    c = pd.concat([c1, c2], axis=1)
                    f.columns=['Train','Test']
                    print(c)
            print('--------------------------------------------')

var_desc('object',alldata)
# var_desc('float64', alldata)
# var_desc('int64', alldata)

## convert obj to num
for c in alldata.columns:
    if alldata[c].dtype=='object' and c not in ['sub_area','timestamp','Source']:
        if len(alldata[c].value_counts())==2:
            print(c)
            alldata['num_'+c]=[0 if x in ['no','OwnerOccupier'] else 1 for x in alldata[c]]
        if len(alldata[c].value_counts())==5:
            print(c)
            alldata['num_'+c]=0
            alldata['num_'+c].loc[alldata[c]=='poor']=0
            alldata['num_'+c].loc[alldata[c]=='satisfactory']=1
            alldata['num_'+c].loc[alldata[c]=='good']=2
            alldata['num_'+c].loc[alldata[c]=='excellent']=3
            alldata['num_'+c].loc[alldata[c]=='no data']=1

## missing values
missing_col = [[c,sum(alldata[alldata.Source=='Train'][c].isnull()==True),sum(alldata[alldata.Source=='Test'][c].isnull()==True)] for c in alldata.columns]
missing_col = pd.DataFrame(missing_col,columns=['Var','missingTrain','missingTest'])

missingdf=missing_col[missing_col.missingTrain+missing_col.missingTest>0]
missingdf=missingdf.sort('missingTrain')
f, ax = plt.subplots(figsize=(6, 15))
sns.barplot(y=missingdf.Var,x=missingdf.missingTrain)

# First, we group variables into small categories,
# Then apply PCA on each of the categories and show correlation plots
excl_col=['id','timestamp','sub_area'] + [c for c in alldata.columns if alldata[c].dtype=='object']
resv_col=['price_doc','LogAmt','Source','cafe_sum_500_max_price_avg','cafe_sum_500_min_price_avg','cafe_avg_price_500','hospital_beds_raion']

def sel_grp(keys):
    lst_all = list()
    for k in keys:
        lst = [c for c in alldata.columns if c.find(k)!=-1 and c not in excl_col and c not in resv_col]
        lst = list(set(lst))
        lst_all += lst
    return(lst_all)

col_grp = dict({})
col_grp['people']=sel_grp(['_all','male'])
col_grp['id'] = sel_grp(['ID_'])
col_grp['church']=sel_grp(['church'])
col_grp['build']=sel_grp(['build_count_'])
col_grp['cafe']=sel_grp(['cafe_count'])
col_grp['cafeprice']=sel_grp(['cafe_sum','cafe_avg'])
col_grp['km']=sel_grp(['_km','metro_min','_avto_min','_walk_min','_min_walk'])
col_grp['mosque']=sel_grp(['mosque_count'])
col_grp['market']=sel_grp(['market_count'])
col_grp['office']=sel_grp(['office_count'])
col_grp['leisure']=sel_grp(['leisure_count'])
col_grp['sport']=sel_grp(['sport_count'])
col_grp['green']=sel_grp(['green_part'])
col_grp['prom']=sel_grp(['prom_part'])
col_grp['trc']=sel_grp(['trc_count'])
col_grp['sqm']=sel_grp(['_sqm_'])
col_grp['raion']=sel_grp(['_raion'])
col_grp['macro']=sel_grp(['mac__'])
col_grp.keys()

col_tmp = list()
for d in col_grp:
    col_tmp+=(col_grp[d])
col_grp['other']=[c for c in alldata.columns if c not in col_tmp and c not in excl_col and c not in resv_col]
col_grp['other']  ## these 'other' variables are not to be PCA

macro_missing_2 = pd.DataFrame([[c,sum(alldata[c].isnull())] for c in col_grp['macro']],columns=['Var','Missing'])
macro_missing_3=macro_missing_2[macro_missing_2.Missing>5000]
print(macro_missing_3)
excl_col+=list(macro_missing_3.Var)
print(excl_col)

# Rebuild the macro group with the new exclusion list
col_grp['macro']=sel_grp(['mac__'])

loopkeys=list(col_grp.keys())
print(loopkeys)

def partial_pca(var,data,col_grp):
    from sklearn.decomposition import PCA
    import bisect
    pca = PCA()
    df = data[col_grp[var]].dropna()
    print([len(data[col_grp[var]]), len(df)])
    df = (df-df.mean())/df.std(ddof=0)
    pca.fit(df)
    varexp = pca.explained_variance_ratio_.cumsum()
    cutoff = bisect.bisect(varexp, 0.95)
    #print(cutoff)
    #print(pca.explained_variance_ratio_.cumsum())
    newcol=pd.DataFrame(pca.fit_transform(X=df)[:,0:(cutoff+1)],columns=['PCA_'+var+'_'+str(i) for i in range(cutoff+1)],index=df.index)
    #print(newcol)
    col_grp['PCA_'+var]=list(newcol.columns)
    return(newcol,col_grp,pca)


# Calculate pca for each grp and append the most important features (95% variance) for each grp to alldata
for c in loopkeys:
    if c!='other':
        print(c)
        newcol,col_grp,pca = partial_pca(c,alldata,col_grp)
        # Add PCA data to alldata, note that the PCA was performed on null-removed data
        # the returned PCA transformation has smaller number of rows than alldata
        alldata = alldata.join(newcol)
        print(alldata.shape)

alldata.to_csv('t.csv')

wpca = list()
wopca = list()
for c in col_grp.keys():
    if c.find('PCA_') != -1:
        wpca += col_grp[c]
    else:
        wopca += col_grp[c]

wpca += col_grp['other']
wpca += resv_col
wopca += col_grp['other']
wopca += resv_col

wpca = list(set(wpca))
wopca = list(set(wopca))

wpca.sort()
wopca.sort()

## Correlation without PCA
corrmat = alldata[wopca].corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=False,yticklabels=False,cbar=False,annot=False);

## Correlation with PCA
corrmat = alldata[wpca].corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=True,yticklabels=True,cbar=False,annot=False);

## Top 20 correlated variables
corrmat = alldata[wpca].corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price_doc')['price_doc'].index
cm = alldata[cols].corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

## these are the variables going into the model.
alldata[wpca].columns

## Add a few more features suggested by other discussions
##
# Add month-year
month_year = (alldata.timestamp.dt.month + alldata.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
alldata['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (alldata.timestamp.dt.weekofyear + alldata.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
alldata['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
alldata['month'] = alldata.timestamp.dt.month
alldata['dow'] = alldata.timestamp.dt.dayofweek

# Other feature engineering
alldata['rel_floor'] = alldata['floor'] / alldata['max_floor'].astype(float)
alldata['rel_kitch_sq'] = alldata['kitch_sq'] / alldata['full_sq'].astype(float)

wpca +=['month_year_cnt','week_year_cnt','dow','month','rel_floor','rel_kitch_sq']
wopca+=['month_year_cnt','week_year_cnt','dow','month','rel_floor','rel_kitch_sq']
allfeature=list(set(wpca+wopca))

## let's try a 5-fold CV
from sklearn.model_selection import KFold
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
kf = KFold(5,shuffle =True)

xgb_params = {
    'eta': 0.05,
    'max_depth': 12,
    'subsample': 1,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'min_child_weight': 200
}

def cv_xgb(val_train_X,val_train_Y,val_val_X,val_val_Y):
    dtrain = xgb.DMatrix(val_train_X, val_train_Y, feature_names=val_train_X.columns)
    dval = xgb.DMatrix(val_val_X, val_val_Y, feature_names=val_val_X.columns)

    # Uncomment to tune XGB `num_boost_rounds`
    partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                           early_stopping_rounds=50, verbose_eval=20)

    num_boost_round = partial_model.best_iteration
    return(num_boost_round,partial_model.best_score)

train_col = [c for c in alldata[wpca].columns if c not in ['price_doc','Source']]
alldata_1 = alldata[alldata.Source=='Train'][train_col]

for val_train, val_val in kf.split(alldata_1):
    val_train_X = alldata_1.ix[val_train].drop('LogAmt',axis=1)
    val_train_Y = alldata_1.ix[val_train].LogAmt
    val_val_X = alldata_1.ix[val_val].drop('LogAmt',axis=1)
    val_val_Y = alldata_1.ix[val_val].LogAmt
    print("%s %s %s %s" % (val_train_X.shape, val_train_Y.shape, val_train.shape, val_val.shape))
    print(cv_xgb(val_train_X,val_train_Y,val_val_X,val_val_Y))
    break  ## this takes long to run, I am breaking it to demonstrate; comment the line if you want full CV

## Xgboost accepts features that are not presented in the sparse feature matrix, null's are treated as 'missing'.
#
#  XGBoost will handle it internally
## Run it on the full training data set
num_boost_round = 200
all_train_X = alldata_1.drop('LogAmt',axis=1)
all_train_Y = alldata_1.LogAmt
all_test_X = alldata[alldata.Source=='Test'][train_col].drop('LogAmt',axis=1)
dtrain_all = xgb.DMatrix(all_train_X, all_train_Y, feature_names=all_train_X.columns)
dtest      = xgb.DMatrix(all_test_X, feature_names=all_test_X.columns)
model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)

## important features
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
xgb.plot_importance(model, height=0.5, ax=ax)

## Make a predicition
ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1
id_test = alldata[alldata.Source=='Test'].id
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('sub_pca.csv', index=False)