#!/usr/bin/python

#Inspect earth surface temperature data
#Identify missing entries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#uncomment for full pandas table
#pd.set_option('display.max_rows', 9999, 'display.max_columns', 12)

#parse data
temp_global = pd.read_csv('GlobalTemperatures.csv',parse_dates=[0], infer_datetime_format=True)
#temp_city = pd.read_csv('GlobalLandTemperaturesByCity.csv',parse_dates=[0], infer_datetime_format=True)
#temp_major_city = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv',parse_dates=[0], infer_datetime_format=True)
#temp_country = pd.read_csv('GlobalLandTemperaturesByCountry.csv',parse_dates=[0], infer_datetime_format=True)
#temp_state = pd.read_csv('GlobalLandTemperaturesByState.csv',parse_dates=[0], infer_datetime_format=True)

#print overview of data
print temp_global.tail(10)
print temp_global.describe()

#drop NaN from before 1800
temp_global.dropna(axis=0, inplace=True)

#set datetime as index
temp_global.set_index(temp_global['dt'],inplace=True)

#plot average over 1 year. 
temp_global.resample('AS').mean().plot()
plt.show()
