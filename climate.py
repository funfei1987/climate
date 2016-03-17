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
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sklearn.cluster import AgglomerativeClustering

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

def hierarchicalCluster(corr_matrix_df, n_clusters):
	"""calculate clustering from the correlation matrix using the hierarchical Ward method"""
	#set method
	ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',affinity='euclidean')

	result=ward.fit_predict(corr_matrix_df)
	cluster_df=pd.DataFrame(result, index=corr_matrix_df.index, columns= ['Cluster'])

	return cluster_df

def plotWorld(city_location_df, color='bo', markersize=18):
	"""plot locations on world map"""

	#convert coordinates
	lons=[]; lats=[]
	for coord in city_location_df['Longitude'].values:
		lons=np.append(lons,conversion(coord))

	for coord in city_location_df['Latitude'].values:
		lats=np.append(lats,conversion(coord))

	#initialize map
	map = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,
		llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
 
 	#map details
	map.drawcoastlines()
	map.drawcountries()
	map.fillcontinents(color = 'coral')
	map.drawmapboundary()

	map.drawparallels(np.arange(-80,81,20),labels=[1,1,0,0])
	map.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1])

	#convert to x-y coordinates
	x,y = map(lons,lats)
	
	world=map.plot(x, y, color, markersize)
	#cbar = map.colorbar(world,location='bottom',pad="5%")

def conversion(old):
	"""convert lon/lat coordinates with directions"""

	direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}
	return (float(old[:-1])) * direction[old[-1]]
	


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


#plot crosscorrelation matrix
# sns.heatmap(temp_corr_df, vmax=1,
#             square=True, xticklabels=5, yticklabels=5,
#             linewidths=.5)
# plt.show()


#use ward hierarchical clustering 
n_clusters=8
cluster_df=hierarchicalCluster(temp_corr_df, n_clusters)

#plot each cluster in world map with a different color
colors=['bo', 'go', 'ro', 'co' , 'mo' , 'yo' ,'ko' ,'wo']

for i in xrange(n_clusters):
	latitude=[]; longitude=[]
	#get list of cities for each cluster
	temp_city= cluster_df[cluster_df['Cluster']==i].index
	
	#get coordinates for each city
	for city in temp_city:
	
		latitude=np.append(latitude,temp_major_city[temp_major_city['City']==city]['Latitude'].iloc[0])
		longitude=np.append(longitude,temp_major_city[temp_major_city['City']==city]['Longitude'].iloc[0])

	city_location_df=pd.DataFrame({'City': temp_city , 'Latitude': latitude, 'Longitude' :longitude})

	#print locations onto world map
	plotWorld(city_location_df, colors[i])

plt.show()


