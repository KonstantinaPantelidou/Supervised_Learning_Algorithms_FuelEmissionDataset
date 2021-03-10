from sklearn import linear_model,datasets,metrics,model_selection
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn import linear_model,datasets,metrics,model_selection
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import math

df = pd.read_csv("fuel_emissions.csv", nrows=2000)
print(df)
print(df.dtypes)

#Converting categorical variable into dummy/indicator variables
df = pd.get_dummies(df,  columns=['file', 'manufacturer',
	'model', 'description',  'transmission',
	'transmission_type', 'fuel_type'])

print(df)
print(df.dtypes)
print(df['fuel_cost_12000_miles'])

#filling NA/NaN values
df=df.fillna(0)
print(df)
print(df.dtypes)
print(df['fuel_cost_12000_miles'])


y=df.fuel_cost_12000_miles
print(y)

X = df.drop('fuel_cost_12000_miles', axis=1)
print(X)

print("shape prin pca")
print(X.shape)
#applying PCA for dimensionality reduction
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
print("shape meta apo pca")
print(X.shape)


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)


#scaling data
scaler = StandardScaler().fit_transform(x_train)
print("scaler")
print(scaler)

min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)


################ Linear Regression ################

linearRegressionModel = LinearRegression()
linearRegressionModel.fit(x_train, y_train)
y_predicted = linearRegressionModel.predict(x_test)

print("###### Linear Regression ######")
print()
print("Correlation:")
print(stats.pearsonr(y_predicted, y_test))
print("MSE:")
print(metrics.mean_squared_error(y_predicted, y_test))
print("R2:")
print(metrics.r2_score(y_predicted, y_test))
print("MAE:")
print(metrics.mean_absolute_error(y_predicted, y_test))
# as MAPE < 10% is Excellen
print("MAPE:")
print(metrics.mean_absolute_percentage_error(y_predicted, y_test))
print("RMSE:")
mse = metrics.mean_squared_error(y_predicted, y_test)
rmse = math.sqrt(mse)
print(rmse)
print()

################ Logistic Regression ################

LogisticRegressionModel = LogisticRegression(max_iter=100)
LogisticRegressionModel.fit(x_train, y_train)
y_predicted = LogisticRegressionModel.predict(x_test)

print("###### Logistic Regression ######")
print()
print("Correlation:")
print(stats.pearsonr(y_predicted, y_test))
print("MSE:")
print(metrics.mean_squared_error(y_predicted, y_test))
print("R2:")
print(metrics.r2_score(y_predicted, y_test))
print("MAE:")
print(metrics.mean_absolute_error(y_predicted, y_test))
# as MAPE < 10% is Excellen
print("MAPE:")
print(metrics.mean_absolute_percentage_error(y_predicted, y_test))
print("RMSE:")
mse = metrics.mean_squared_error(y_predicted, y_test)
rmse = math.sqrt(mse)
print(rmse)
print()
################ Ridge Regression ################

RidgeRegressionModel = Ridge(alpha=.5)
RidgeRegressionModel.fit(x_train, y_train)
y_predicted = RidgeRegressionModel.predict(x_test)

print("###### Ridge Regression ######")
print()
print("Correlation:")
print(stats.pearsonr(y_predicted, y_test))
print("MSE:")
print(metrics.mean_squared_error(y_predicted, y_test))
print("R2:")
print(metrics.r2_score(y_predicted, y_test))
print("MAE:")
print(metrics.mean_absolute_error(y_predicted, y_test))
# as MAPE < 10% is Excellen
print("MAPE:")
print(metrics.mean_absolute_percentage_error(y_predicted, y_test))
print("RMSE:")
mse = metrics.mean_squared_error(y_predicted, y_test)
rmse = math.sqrt(mse)
print(rmse)
print()
################ Lasso Regression ################

LassoRegressionModel = Lasso()
LassoRegressionModel.fit(x_train, y_train)
y_predicted = LassoRegressionModel.predict(x_test)

print("###### Lasso Regression ######")
print()
print("Correlation:")
print(stats.pearsonr(y_predicted, y_test))
print("MSE:")
print(metrics.mean_squared_error(y_predicted, y_test))
print("R2:")
print(metrics.r2_score(y_predicted, y_test))
print("MAE:")
print(metrics.mean_absolute_error(y_predicted, y_test))
# as MAPE < 10% is Excellen
print("MAPE:")
print(metrics.mean_absolute_percentage_error(y_predicted, y_test))
print("RMSE:")
mse = metrics.mean_squared_error(y_predicted, y_test)
rmse = math.sqrt(mse)
print(rmse)
print()
