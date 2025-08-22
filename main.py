import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df=pd.read_csv('housing.csv')

# print(df.head())
# print(df.shape)
# print(df.isnull().sum())

df=df.dropna()
# print(df.isnull().sum())
# print(df.shape)

df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

x=df.drop('median_house_value',axis=1)
y=df['median_house_value']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=2)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0]
}

model = XGBRegressor(random_state=2)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
grid_search.fit(x_train,y_train)

print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred=best_model.predict(x_test)

print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
r2_score = metrics.r2_score(y_test, y_pred)
print(f"R-squared: {r2_score}")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns, 'model_columns.pkl')
