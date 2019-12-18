#------------------------------ ----------------------
#-------------------Libraries ---------------
#------------------------------ ----------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
#------------------------------ ----------------------
#-------------------Data_imshow ---------------
#------------------------------ ----------------------
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_all = pd.concat([data_train, data_test])
#data_train.SalePrice.describe()
plt.hist(data_train.SalePrice, color='red')
plt.show()
#------------------------------ ----------------------
#-------------------Data_preprocessing ---------------
#------------------------------ ----------------------

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

#-------------------------- Features_modify------------------------------------

columns = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')

for c in columns:
    lbl = LabelEncoder() 
    lbl.fit(list(data_all[c].values)) 
    data_all[c] = lbl.transform(list(data_all[c].values))

#-------------------------- Preparing data ------------------------------------
        
train_dataset = data_all[0:1460]
test_dataset = data_all[1461:2919]

#Take targate variable into y
y = train_dataset['SalePrice']

del train_dataset['SalePrice']

X = train_dataset.values
y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


model = linear_model.LinearRegression()

model.fit(X_train, y_train)

print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))

print("Accuracy --> ", model.score(X_test, y_test)*100)


GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, y_train)
print("Accuracy GBR--> ", GBR.score(X_test, y_test)*100)
