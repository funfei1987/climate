#!/usr/bin/python

#Analysis of earth surface temperature
#Data from Kaggl.com (originally from Berkeley Earth data page)

import pickle
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# ******  FUNCTIONS  ******

def linFit(temp):
	"""perfom linear fit on temperature data for each city"""

	#make series of cities
	temp_city = temp_major_city['City'].unique()
	slope_array=[]; r_array=[]; p_array=[]; std_array=[]

	for i in xrange(len(temp_city)):
		#transform df to np.array
		temp_data=temp.loc[temp['City']==temp_city[i]]['AverageTemperature'].values.astype(float)
		time_data=temp.loc[temp['City']==temp_city[i]].index.values.astype('datetime64[M]').astype(float)

		#linreg
		slope, intercept, r_value, p_value, std_err = sp.stats.linregress(time_data,temp_data)

		slope_array=np.append(slope_array, slope)
		r_array=np.append(r_array, r_value)
		p_array=np.append(p_array, p_value)
		std_array=np.append(std_array, std_err)

	#create DataFrame from results
	temp_diff =  pd.DataFrame({'City' : temp_city, 'slope' : slope_array, 'r2' : r_array**2 , 'p' : p_array  , 'std_error' : std_array  } )
	return temp_diff



##########################################################################################

#parse data from pickled binary
with  open('temp_major_city.pkl', 'rb') as handle:
	temp_major_city = pickle.load(handle)

temp_major_city.set_index(temp_major_city['dt'],inplace=True)

#drop dates before 1960 and rows with NaN
temp_major_city = temp_major_city[(temp_major_city.index.year >= 1960)]
temp_major_city = temp_major_city.dropna()


#perfom linear regression on all the cities 
temp_diff = linFit(temp_major_city)

print temp_diff
print temp_diff.describe()


#print temp_major_city.loc[temp_major_city['City']=='Abidjan']
