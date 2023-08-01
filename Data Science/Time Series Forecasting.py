#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('dc.csv')

df = df.rename(columns = {'Unnamed: 0' : 'Time'})
df['Time'] = pd.to_datetime(df['Time'])
df = df.iloc[::-1].set_index('Time')

df.head()


# In[2]:


import matplotlib.pyplot as plt

plt.plot(df['close_USD'])
plt.show()


# In[3]:


# Split the data
train = df.iloc[:-200] 
test = df.iloc[-200:]


# In[10]:


# Fit model on training data 

endog = train['close_USD'] 
model = ARIMA(endog, order=(2, 1, 0))
results = model.fit()

# Make predictions on test set
start = len(train)
end = len(train) + len(test) - 1
forecast = results.predict(start=start, end=end)
forecast


# In[17]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Extract actual values from test set 
actual = test['close_USD']

# Calculate errors
mae = mean_absolute_error(actual, forecast)
mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) * 100

print('MAE: %.3f' % mae)
print('RMSE: %.3f' % rmse)
print('MAPE: %.2f%%' % mape)


# In[18]:


plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.show()


# In[26]:


df_p = df.reset_index()[["Time", "close_USD"]].rename(columns={"Time": "ds", "close_USD": "y"})


# In[20]:


import pandas as pd
from prophet import Prophet

model = Prophet()

# Fit the model
model.fit(df_p)

# create date to predict
future_dates = model.make_future_dataframe(periods=365)

# Make predictions
predictions = model.predict(future_dates)

predictions.head()


# In[21]:


model.plot(predictions)


# In[22]:


model.plot_components(predictions)


# In[23]:


from prophet.diagnostics import cross_validation, performance_metrics

# Perform cross-validation with initial 365 days for the first training data and the cut-off for every 180 days.

df_cv = cross_validation(model, initial='365 days', period='180 days', horizon = '365 days')

# Calculate evaluation metrics
res = performance_metrics(df_cv)

res


# In[24]:


from prophet.plot import plot_cross_validation_metric
#choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'

plot_cross_validation_metric(df_cv, metric= 'mape')

