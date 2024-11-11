### DEVELOPED BY: AJAY ASWIN M
### REGISTER NO: 212222240005
### DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the Petrol Price dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('petrol.csv')
data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].dt.date
petrol_price = data.groupby('Date')['Chennai'].sum().reset_index()

petrol_price['Date'] = pd.to_datetime(petrol_price['Date'])
petrol_price.set_index('Date', inplace=True)

plt.plot(petrol_price.index, petrol_price['Chennai'])
plt.xlabel('Date')
plt.ylabel('Petrol Price ($)')
plt.title('Petrol Price Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(petrol_price['Chennai'])

plot_acf(petrol_price['Chennai'])
plt.show()
plot_pacf(petrol_price['Chennai'])
plt.show()

train_size = int(len(petrol_price) * 0.8)
train, test = petrol_price['Chennai'][:train_size], petrol_price['Chennai'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Petrol Price ($)')
plt.title('SARIMA Model Predictions for Petrol Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()


```
### OUTPUT:
![image](https://github.com/user-attachments/assets/18588c7f-8cd6-4df7-a88d-7960ef2a612b)
![image](https://github.com/user-attachments/assets/e69ddd5b-ce11-4cad-86fc-8bf52d4984ac)
![image](https://github.com/user-attachments/assets/b8ad29ed-8c3d-4359-bf7f-db812264a77a)
![image](https://github.com/user-attachments/assets/b52cdec2-15ef-44c2-913f-5a9a29dc343a)


### RESULT:
Thus the program run successfully based on the SARIMA model.
