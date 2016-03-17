#!/usr/bin/python

#Inspect earth surface temperature data
#Identify missing entries

import pickle
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


#uncomment for full pandas table
#pd.set_option('display.max_rows', 9999, 'display.max_columns', 12)

#parse data
#temp_global = pd.read_csv('GlobalTemperatures.csv',parse_dates=[0], infer_datetime_format=True)
#temp_city = pd.read_csv('GlobalLandTemperaturesByCity.csv',parse_dates=[0], infer_datetime_format=True)
#temp_major_city = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv',parse_dates=[0], infer_datetime_format=True)
#temp_country = pd.read_csv('GlobalLandTemperaturesByCountry.csv',parse_dates=[0], infer_datetime_format=True)
#temp_state = pd.read_csv('GlobalLandTemperaturesByState.csv',parse_dates=[0], infer_datetime_format=True)

#load and safe data with pickle
# with  open('temp_major_city.pkl', 'wb') as handle:
	# pickle.dump(temp_major_city, handle)
with  open('temp_major_city.pkl', 'rb') as handle:
	temp_major_city = pickle.load(handle)

#set datetime as index
temp_major_city.set_index(temp_major_city['dt'],inplace=True)
#drop results before 1960 and NaN
temp_major_city = temp_major_city[(temp_major_city.index.year >= 1960)]
temp_major_city = temp_major_city.dropna()

temp=temp_major_city

temp_diff = pd.Series(data=None, index=temp_major_city['City'].unique()) 

#select city

inputCity='London'
#transform df to np.array
temp_data=temp.loc[temp['City']==inputCity]['AverageTemperature'].values.astype(float)
time_data=temp.loc[temp['City']==inputCity].index.values.astype('datetime64[M]').astype(float)

#linreg
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(time_data,temp_data)
print(slope, r_value**2, p_value, std_err)
#plot
sns.regplot(time_data,temp_data)
plt.show()


#for Temps
#temp_global.loc[:,['LandAverageTemperature', 'LandMaxTemperature','LandMinTemperature','LandAndOceanAverageTemperature']].resample('AS').mean().plot(color=['b', 'r','y', 'k'], title = 'Temperatures')




