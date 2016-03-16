#!/usr/bin/python

#Analysis of earth surface temperature
#Data from Kaggl.com (originally from Berkeley Earth data page)

import importlib
import sys
reload(sys)
sys.setdefaultencoding('utf8')

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
	temp_city = temp['City'].unique()
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


def correl(df,city1, city2):
    """calculates correlations of AverageTemperature between city1 and city 2 in df"""
    
    return(df.loc[df['City'] == city1 ]['AverageTemperature'].corr(df.loc[df['City'] == city2 ]['AverageTemperature']))

def correl_array(temp):
    """calculates correlations of AverageTemperature between all pairs of cities"""
    #make series of cities from temp and empty correlation array
    temp_city = temp['City'].unique()
    temp_corr_array = []
    
    #loop over the cities and calculate their correlations
    for city1 in temp_city:
        for city2 in temp_city:
            temp_corr_array = np.append(temp_corr_array,correl(temp,city1, city2))
    #reshape and form dataframe with index, columns and len given by series of cities 
    temp_corr_df = pd.DataFrame(np.reshape(temp_corr_array, (len(temp_city), len(temp_city))),index = temp_city ,columns = temp_city)
    return temp_corr_df


##########################################################################################


#parse data from pickled binary
with  open('temp_major_city.pkl', 'rb') as handle:
	temp_major_city = pickle.load(handle)

temp_major_city.set_index(temp_major_city['dt'],inplace=True)

#drop dates before 1960 and rows with NaN
temp_major_city = temp_major_city[(temp_major_city.index.year >= 1960)]
temp_major_city = temp_major_city.dropna()


#perfom linear regression on AverageTemperature for all the cities
temp_diff = linFit(temp_major_city)

#calculate correlation of AverageTemperature between all pairs of cities
temp_corr_df = correl_array(temp_major_city)


print temp_diff.head()
print temp_diff.describe()

print temp_corr_df.head()
print temp_corr_df.describe()



#print temp_major_city.loc[temp_major_city['City']=='Abidjan']
