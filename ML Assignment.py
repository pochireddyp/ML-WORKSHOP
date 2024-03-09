#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()


# In[1]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='red', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='yellow', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()


# In[5]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='brown', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='green', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()


# In[6]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)


# In[7]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)


# In[15]:


'''
Developed by : K MADHAVA REDDY
Register Number : 212223240064
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.1, 0.4, 0.5, 0.8]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')


# In[ ]:




