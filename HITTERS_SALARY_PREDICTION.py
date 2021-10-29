import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Lasso,Ridge
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(df,col,q1=0.10,q3=0.90):
    q1 = df[col].quantile(q1)
    q3 = df[col].quantile(q3)
    iqr_range = q3 - q1
    up_limit = q3 + 1.5 * iqr_range
    low_limit = q1 - 1.5 * iqr_range
    return low_limit, up_limit

def check_outlier(df,col):
    low_limit, up_limit =outlier_thresholds(df,col)
    if df[(df[col] < low_limit) |  (df[col] > up_limit)].any(axis=None):
        return True
    else:
        return False

def remove_outlier(df,col):
    low_limit,up_limit = outlier_thresholds(df,col,0.20,0.80)
    df_without_outliers = df[~((df[col] < low_limit) | (df[col] > up_limit))]
    return df_without_outliers

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted)/actual))*100

# uç değer kontrolü yaparken median ve ortalaması arasında %20'den fazla olanlar için olanları seçebilmek için oluşturuldu
def Mean_Median_Diff(df,num_cols,Ratio=20):
    Mean_Median_Diff_Col = []
    for col in df[num_cols]:
        if df[col].median()<=df[col].mean():
            if (df[col].mean()- df[col].median())/df[col].mean()*100>Ratio:
                print(col,(df[col].mean()- df[col].median())/df[col].mean()*100)
                Mean_Median_Diff_Col.append(col)
        else:
            if (df[col].median()- df[col].mean())/df[col].median()*100>Ratio:
                print(col,(df[col].median()- df[col].mean())/df[col].median()*100)
                Mean_Median_Diff_Col.append(col)
    return Mean_Median_Diff_Col


def high_correlated_cols(df, plot=False, corr_th=0.90):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def ratio_na_col(df):
    Na_Var=[col for col in df.columns if  df[col].isnull().sum()>0]
    for col in df[Na_Var]:
        print("Na Variable : ",col)
        print(df[df[col].isnull()],"\n")
        df[col].fillna(0,inplace=True)

def outlier_rows(df,num_cols,th=-2,plot=False):
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df[num_cols])
    df_scores = clf.negative_outlier_factor_
    scores = pd.DataFrame(np.sort(df_scores))
    if plot:
        scores.plot(stacked=True, xlim=[0, 20], style='.-')
        plt.show()
    print(scores[:10])
    df.drop(axis=0,labels= df[df_scores < th].index,inplace=True)

df_ = pd.read_csv(r"C:\Users\HP\Desktop\HITTERS_SALARY_PREDICTION\source_file\hitters.csv")
df_.isnull().sum()
df_.shape
df = df_.dropna().copy()
df.head()
df.describe().T

#df.groupby("Years").agg({"Salary":["mean","min","max","count"]})
#df.loc[df["Years"]==1,"Salary"].sort_values(ascending=False).head(20)


df.loc[(df['Years'] < 7), 'New_Years_Big_7_Flag'] = 0
df.loc[(df['Years'] >= 7), 'New_Years_Big_7_Flag'] =1 #7 seneden sonra maaşlarda bir düşüş söz konusu

df["New_AtBat_Mean"] =(df["CAtBat"]/df["Years"])
df["New_Hits_Mean"] =(df["CHits"]/df["Years"])
df["New_HmRun_Mean"] =(df["CHmRun"]/df["Years"])
df["New_Runs_Mean"] =(df["CRuns"]/df["Years"])
df["New_RBI_Mean"] =(df["CRBI"]/df["Years"])
df["New_Walks_Mean"] =(df["CWalks"]/df["Years"])

df.loc[(df['AtBat'] >= df['New_AtBat_Mean']), 'New_AtBat_86_Big'] = 1
df.loc[(df['AtBat']< df['New_AtBat_Mean']), 'New_AtBat_86_Big'] = 0

df.loc[(df['Hits'] >= df['New_Hits_Mean']), 'New_Hits_86_Big'] = 1
df.loc[(df['Hits']< df['New_Hits_Mean']), 'New_Hits_86_Big'] = 0

df.loc[(df['HmRun'] >= df['New_HmRun_Mean']), 'New_HmRun_86_Big'] = 1
df.loc[(df['HmRun']< df['New_HmRun_Mean']), 'New_HmRun_86_Big'] = 0

df.loc[(df['Runs'] >= df['New_Runs_Mean']), 'New_Runs_86_Big'] = 1
df.loc[(df['Runs']< df['New_Runs_Mean']), 'New_Runs_86_Big'] = 0

df.loc[(df['RBI'] >= df['New_RBI_Mean']), 'New_RBI_86_Big'] = 1
df.loc[(df['RBI']< df['New_RBI_Mean']), 'New_RBI_86_Big'] = 0

df.loc[(df['Walks'] >= df['New_Walks_Mean']), 'New_Walks_86_Big'] = 1
df.loc[(df['Walks']< df['New_Walks_Mean']), 'New_Walks_86_Big'] = 0

df["Score"]=(df["New_AtBat_86_Big"] + df["New_Hits_86_Big"] *0.05 + df["New_HmRun_86_Big"]*0.40  + \
df["New_Runs_86_Big"] *0.30+ df["New_Walks_86_Big"] *0.10 + df["New_RBI_86_Big"]*0.15 ) *df["Years"]

df["New_Hits_Per"] =(df["Hits"] / df["AtBat"]) * 100
df["New_HmRun_Per"] =(df["HmRun"]/df["AtBat"])*100
df["New_HmRun_Hits_Per"] =(df["HmRun"]/df["Hits"])*100
df["New_HmRun_Runs_Per"] =(df["HmRun"]/df["Runs"])*100


df["New_Per_Sum"] =(df["New_Hits_Per"]+df["New_HmRun_Per"] +df["New_HmRun_Hits_Per"]+df["New_HmRun_Runs_Per"])*df["Score"]

ratio_na_col(df)
#df.isnull().sum()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

outlier_rows(df,num_cols,th=-2)
df.shape

Mean_Median_Diff_Col = Mean_Median_Diff(df, num_cols)
for col in df[Mean_Median_Diff_Col].columns:
    df = remove_outlier(df, col)
df.shape

df["Salary"].sort_values(ascending=False).head(50)
df=df.loc[df["Salary"]<1300]
df.groupby("Years").agg({"Salary":["mean","min","max","count"]})


df[cat_cols].nunique()
dms = pd.get_dummies(df[cat_cols])
dms_ratio = pd.DataFrame()
for col in dms.columns:
    dms_ratio[col]=(dms[col].value_counts()/len(dms)*100)
dms_ratio


y=df["Salary"]
dms = pd.get_dummies(df[cat_cols],drop_first=True).astype("int64")
X_ =df.drop(["Salary"]+cat_cols,axis=1)
X = pd.concat([X_,dms],axis=1)
y.describe()
X.describe().T



dict_corr ={col:X[col].corr(y) for col in X.columns}
dict_corr=dict(sorted(dict_corr.items(), key=lambda x: abs(x[1]), reverse=True))
dict_corr

######################################################
# Base Models
######################################################
models = [('LR', LinearRegression()),
          ("Ridge", Ridge(random_state=22)),
          ("Lasso", Lasso(random_state=22)),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor(random_state=22)),
          ('RF', RandomForestRegressor(random_state=22)),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor(random_state=22)),
          ("XGBoost", XGBRegressor(objective='reg:squarederror',random_state=22)),
          ("LightGBM", LGBMRegressor(random_state=22)),
          ("CatBoost", CatBoostRegressor(verbose=False,random_state=22))
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    print(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))

######################################################
# Automated Hyperparameter Optimization
######################################################
cart_params = {'max_depth': range(1, 6),
               "min_samples_split": range(2, 30)}
rf_params = {"max_depth": [6, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [500, 1000,2000]}
xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5,6],
                  "n_estimators": [200, 300, 500],
                  "colsample_bytree": [0.5, 0.7]}
lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [500, 1500],
                   "colsample_bytree": [0.5, 0.7]}
catb_params = {"iterations":[200,500,750],
               "learning_rate": [0.01,0.1,0.001],
               "depth": [5,8]}
regressors = [("CART", DecisionTreeRegressor(random_state=22), cart_params),
              ("RF", RandomForestRegressor(random_state=22), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror',random_state=22), xgboost_params),
              ('LightGBM', LGBMRegressor(random_state=22), lightgbm_params),
              ('CATB',CatBoostRegressor(verbose=False,random_state=22),catb_params)]
best_models = {}
for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
    print(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
    best_models[name] = final_model

######################################################
# # Stacking & Ensemble Learning
######################################################
voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('XGBoost', best_models["XGBoost"]),
                                         ('LightGBM', best_models["LightGBM"]),
                                         ('CATB', best_models["CATB"])])
voting_reg.fit(X, y)
np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
# RMSE 156.2695117417943
# cross_val_RMSE = [163.30755133, 104.91479919, 184.02472808, 126.05285723,
#        175.09017118, 171.67433852, 135.42146292, 170.43037207,
#        188.09769757, 143.68113933])

X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)



