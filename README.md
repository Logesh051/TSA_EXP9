# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 4/11/2025
### Name: LOGESH N A
### Reg:  212223240078

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# load dataset
df = pd.read_csv("/content/NFLX.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# arima model function
def arima_model(df, col, order):

    # split into train and test
    n = int(len(df)*0.8)
    train, test = df[col][:n], df[col][n:]

    # fit model
    model = ARIMA(train, order=order).fit()

    # forecast
    pred = model.forecast(len(test))

    # calculate error
    rmse = np.sqrt(mean_squared_error(test, pred))

    # plot results
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(pred, label="Forecast")
    plt.title(f"ARIMA Forecasting - {col}")
    plt.legend()
    plt.show()

    print("RMSE:", rmse)

# call function
arima_model(df, "Close", (5,1,0))
```
### OUTPUT:
<img width="851" height="579" alt="image" src="https://github.com/user-attachments/assets/8e5ef785-bb43-41b0-9fbb-e27d29ade746" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
