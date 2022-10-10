from sklearn import metrics, linear_model, model_selection
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')

data = np.array(data[['Age', 'DailyRate', 'Education', 'HourlyRate', 'JobLevel', 'MonthlyRate', 'PercentSalaryHike']].values)
print(data)

dt_Train, dt_Test = model_selection.train_test_split(data, test_size=0.3, shuffle=False)

x_test, y_test = dt_Test[:,:-1], dt_Test[:,-1:]

kfold = model_selection.KFold(shuffle=False, random_state=None, n_splits=3)

min = 999999999999
reg = ''

for i, j in kfold.split(dt_Train):
    x_train = dt_Train[i][:, :-1]
    y_train = dt_Train[i][:, -1:]

    x_val = dt_Train[j][:, :-1]
    y_val = dt_Train[j][:, -1:]

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    
    y_pred_train = regr.predict(x_train)
    y_pred_val = regr.predict(x_val)
    
    error = metrics.mean_absolute_error(y_train, y_pred_train) + metrics.mean_absolute_error(y_val, y_pred_val)
    if(error < min):
        min = error
        reg = regr
        
print("W = ", reg.coef_)
print("W0 = ", reg.intercept_)

data_pred = reg.predict(x_test)

print("^^: ", reg.score(x_test, y_test))