#------------------------------ ----------------------
#-------------------Libraries ---------------
#------------------------------ ----------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#------------------------------ ----------------------
#-------------------Data_imshow ---------------
#------------------------------ ----------------------
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
plt.hist(data_train.SalePrice, color='red')
plt.show()
#------------------------------ ----------------------
#-------------------Data_preprocessing ---------------
#------------------------------ ----------------------
##----------------------- Outliers_cleaning ----------------------
outliers = {"LotArea": 140000, "BsmtFinSF1": 4000, "TotalBsmtSF": 5000, 
            "1stFlrSF": 4000, "GrLivArea": 5000, "GarageYrBlt": 2019}
def outliers_cleaning(data, outliers):
    for i in outliers:
        data = data[data[i] < outliers[i]]
    return data
data_train = outliers_cleaning(data_train, outliers)
data_all = pd.concat([data_train, data_test])
#-----------------------Missing ----------------------
All_missing = data_all.isnull().sum()
Train_missing = data_train.isnull().sum()
data_train.columns[data_train.isnull().any()]
total_cells = np.product(data_test.shape) + np.product(data_train.shape)
total_missing = data_train.isnull().sum() + data_test.isnull().sum()
Per = (total_missing/total_cells) * 100
sns.heatmap(data_all.isnull(),yticklabels=False,cbar=False)

print('----------------------------------------------------------------------')
#---------------------- Fill Missing Values------------------------------------

data_all.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu','Utilities'],axis=1,inplace=True)
data_all['LotFrontage'] = data_all.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    data_all[col] = data_all[col].fillna('None')

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    data_all[col] = data_all[col].fillna(int(0))

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    data_all[col] = data_all[col].fillna('None')
    
data_all['MasVnrArea'] = data_all['MasVnrArea'].fillna(int(0))
data_all['MasVnrType'] = data_all['MasVnrType'].fillna('None')

data_all['Electrical'] = data_all['Electrical'].fillna(data_all['Electrical']).mode()[0]
data_all['MSZoning'] = data_all['MSZoning'].fillna(data_all['MSZoning']).mode()[0]
data_all['Functional'] = data_all['Functional'].fillna(data_all['Functional']).mode()[0]
data_all['BsmtHalfBath'] = data_all['BsmtHalfBath'].fillna(data_all['BsmtHalfBath']).mode()[0]
data_all['BsmtFullBath'] = data_all['BsmtFullBath'].fillna(data_all['BsmtFullBath']).mode()[0]
data_all['TotalBsmtSF'] = data_all['TotalBsmtSF'].fillna(data_all['TotalBsmtSF']).mode()[0]
data_all['SaleType'] = data_all['SaleType'].fillna(data_all['SaleType']).mode()[0]
data_all['KitchenQual'] = data_all['KitchenQual'].fillna(data_all['KitchenQual']).mode()[0]
data_all['Exterior2nd'] = data_all['Exterior2nd'].fillna(data_all['Exterior2nd']).mode()[0]
data_all['Exterior1st'] = data_all['Exterior1st'].fillna(data_all['Exterior1st']).mode()[0]
data_all['BsmtFinSF1'] = data_all['BsmtFinSF1'].fillna(data_all['BsmtFinSF1']).mode()[0]
data_all['BsmtFinSF2'] = data_all['BsmtFinSF2'].fillna(data_all['BsmtFinSF2']).mode()[0]
data_all['BsmtUnfSF'] = data_all['BsmtUnfSF'].fillna(data_all['BsmtUnfSF']).mode()[0]

sns.heatmap(data_all.isnull(),yticklabels=False,cbar=False)
N_final = data_all.isnull().sum()

##-------------------------- Features_modify------------------------------------

Col = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
for j in Col:
    lbl = LabelEncoder() 
    lbl.fit(list(data_all[j].values)) 
    data_all[j] = lbl.transform(list(data_all[j].values))
#
##-------------------------- Preparing data ------------------------------------
   
train_dataset = data_all[:len(data_train)]
test_dataset = data_all[len(data_train):]
test_dataset.drop(['SalePrice'],axis=1,inplace=True)

X  = train_dataset.drop(["SalePrice","Id"], axis=1).copy()
y = np.log(train_dataset["SalePrice"])

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
##-------------------------- Prediction ------------------------------------

lr = linear_model.LinearRegression()
model = lr.fit(X_train, Y_train)

print ("R^2:", model.score(X_test, y_test))

predictions = model.predict(X_test)

print ('RMSE:', mean_squared_error(y_test, predictions))

actual_values = y_test

for i in range (-1, 4):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, Y_train)
    preds_ridge = ridge_model.predict(X_test)
    overlay = 'R_sq/RMSE'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
submission = pd.DataFrame()
submission['Id'] = data_test.Id

Res = test_dataset.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

#-------------------------- Final Data Processing and Saving ------------------
predictions = model.predict(Res)
final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions
submission.to_csv('Final_Submussion.csv', index=False)
#------------------------------------------------------------------------------





