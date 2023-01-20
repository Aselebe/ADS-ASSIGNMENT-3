# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:11:08 2023

@author: haselebe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
import err_ranges as err

from sklearn import preprocessing

indicator = 'GDP per capita (current US$)'
year1 = '1989'
year2 = '2021'
url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=excel" 

"""
This is defining a function read_data(url) that takes in a url as an 
argument. The function reads an excel file from the given url using the
 pd.read_excel() function from pandas library. The skiprows=3 argument is
 used to skip the first 3 rows of the file, which usually contain metadata.

The function then drops some unwanted columns from the dataframe using the 
drop() function, specifically the 'Country Code', 'Indicator Name', and 
'Indicator Code' columns.

It creates two dataframes:

df_country which is a dataframe with countries as columns
df_years which is a dataframe with year as columns. This is achieved 
by transposing the dataframe so that columns become rows and vice versa.
The function returns both dataframes as a tuple.

"""

def read_data(url):
    data = pd.read_excel(url, skiprows=3)
    
    data = data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)

    #this extracts a dataframe with countries as column
    df_country = data
    
    #this section extract a dataframe with year as columns
    df_years = data.transpose()

       
    df_years = df_years.rename(columns=df_years.iloc[0])
    df_years = df_years.drop(index=df_years.index[0], axis=0)
    df_years['Year'] = df_years.index
    return df_country, df_years

gdp_country_data, gdp_year_data = read_data(url)


monaco = gdp_country_data.iloc[149,30:]

monaco.plot()
plt.title('MONACO GDP per capita (current US$)')
plt.xlabel('Year')
plt.ylabel('Value in current US$')
plt.show()


Algeria_Switz = gdp_country_data.iloc[[37,60], 30:]

#data=[213,37]
#x=gdp_country_data.iloc[:,data]

#somalia_Switz=gdp_country_data["1988", "1989"]

#plt.title('MONACO GDP per capita (current US$)')
#plt.xlabel('Year')
#plt.ylabel('Value in current US$')
#plt.show()


Algeria_Switz=Algeria_Switz.fillna("0")

soma=Algeria_Switz.transpose()

soma=soma.rename(columns={37:"Algeria",60:"Switzerland" })


soma.plot(kind="line")
plt.title('GAP BETWEEN ALGERIA AND SWITZERLAND')
plt.xlabel('Year')
plt.ylabel('GDP in current US$')
plt.show()


#extract the required data for the clustering
gdp_data = gdp_country_data.loc[gdp_country_data.index, ['Country Name', year1, year2]].dropna()



x = gdp_data[[year1, year2]].dropna().values

"""
The  MinMaxScaler class below from the scikit-learn library's preprocessing module to scale , so that all the values 
are normalized and between 0 and 1. The fit_transform method is used to fit the scaler to the data and then transform it. 
The result of this operation is stored in the variable x_norm. 
This type of scaling is useful when the data has a large range and you want to bring it within a smaller range, like 0 to 1.
"""
scaler=preprocessing.MinMaxScaler()
x_norm=scaler.fit_transform(x)


gdp_data.plot(year1, year2, kind='scatter')
plt.title('Scatter plot ')
plt.xlabel(year1)
plt.ylabel(year2)
plt.show() 

"""
This code is using the "elbow method" to find the optimal number of clusters 
for the K-means algorithm. The elbow method is a heuristic method to determine
 the number of clusters in a dataset.

First, an empty list sse is created. Then, a for loop iterates over the range
 from 1 to 11 (inclusive) and for each iteration, it creates a new KMeans
 object with the number of clusters set to the current iteration value. It 
 also sets some other parameters such as 'k-means++' as the initialization 
 method, max_iter as 300, n_init as 10, and random_state as 0.

It then fits the kmeans model to the data using the fit() method, and appends
 the value of the inertia_ attribute to the sse list. The inertia_ attribute
 returns the sum of squared distances of samples to their closest cluster
 center.

After the loop, it plots the range of number of clusters as x-axis and the 
sse as y-axis. It then sets the title as "Elbow method", x-label as "Number 
of clusters" and y-label as "Inertia" and finally shows the plot.

The idea behind the elbow method is that as the number of clusters increases,
 the SSE decreases. At some point, however, the decrease in SSE will not be 
 proportional to the number of clusters added. This point is the elbow of the
 plot and the number of clusters at this point is considered as the optimal
 number of clusters.

"""

RANGE = range(0, 11)

def plot_vals(vals, RANGE):
  plt.grid()
  plt.xticks(RANGE)
  
vals = [1,2,3,4,6,5,7,8,9,10]

plot_vals(vals, RANGE)


#for i in range(1, 11):
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
    
    
plt.plot(range(1,11), sse, marker="o", color="red")
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)



#this creates a new dataframe with the labels for each country
gdp_data['label'] = y_kmeans


df_label = gdp_data.loc[gdp_data['label'] == 0]
df_label.head(20)

y = kmeans.cluster_centers_

"""
This code is using matplotlib to visualize the results of the K-means 
clustering. It creates a scatter plot with each point representing a country 
and the color of the point representing the cluster to which the country
 belongs. The x-axis represents the GDP per capita of the country in 1989 
 and the y-axis represents the GDP per capita of the country in 2021.

It uses a loop to scatter the data points of each cluster in different colors:
    purple for cluster 1, orange for cluster 2, and green for cluster 3.
It also plots the centroids of each cluster in red.

The plt.legend() function is used to add a legend to the plot to indicate 
which color represents which cluster. Finally, it uses the plt.show() function
 to display the plot.
"""


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'purple',label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'orange',label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green',label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, c = 'red', label = 'Centroids')
plt.legend(loc='upper left', shadow=True, fontsize='small')

plt.title('GDP OF ALL COUNTRIES')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


def model(x, a, b, c, d):
    '''
 This is a function definition for a function named "model" that takes in four
 parameters (a, b, c, d) and one input variable (x). The function has not been
 given a specific task or calculation to perform yet, so it does not have a 
 return statement.
  This function takes in four parameters (a, b, c, d) and one input variable
  (x) and returns the result of the mathematical expression
  "ax**3 + bx2 + c*x + d" where "" denotes exponentiation and "*" 
  denotes multiplication. This is a polynomial equation of degree 3 with the 
  variable x.
 
    '''
    return a*x**3 + b*x**2 + c*x + d

fitting = gdp_year_data[['Year', 'China']].apply(pd.to_numeric,
                                                 errors='coerce')


"""
This code is using the scipy.optimize module to fit a curve to a set of data.
 The curve_fit() function is being used to fit the model function to the data 
 in x_axis and y_axis. The function returns two variables: popt contains the 
 optimal values for the parameters of the model function, and covar contains
 the covariance of the parameters.
"""

fitting.plot("Year", "China", kind="scatter")
plt.title('CHINA GDP per capita (current US$)')
plt.xlabel('Year')
plt.ylabel('Value in current US$')
plt.show()

data_fitting = fitting.dropna().values

x_axis = data_fitting[:,0]
y_axis = data_fitting[:,1]

import numpy as np
"""
It then uses tuple unpacking to assign the optimal values of the parameters
 of the model function (a, b, c, d) to the variables a, b, c and d 
 respectively. It is important to note that the number of variables 
 should match the number of parameters in the model function.
"""

popt, _ = opt.curve_fit(model, x_axis, y_axis)
param, covar = opt.curve_fit(model, x_axis, y_axis)
a, b, c, d = popt


sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x_axis, model, popt, sigma)

x_line = np.arange(min(x_axis), max(x_axis)+1, 1)
y_line = model(x_line, a, b, c, d)

print('low',low, 'up',up)
print(up.shape)


plt.scatter(x_axis, y_axis,color="red")
plt.plot(x_line, y_line, '--', color='black')
plt.title('CHINA CURVE FIT')
plt.xlabel('Year')
plt.ylabel('Value in current US$')
plt.show()

ci=1.96 * np.std(y_axis)/np.sqrt(len(x_axis))
low=y_line-ci
up=y_line+ci

"""
his code is plotting the data in x_axis and y_axis as a scatter plot using 
matplotlib. It then calculates the standard deviation of the parameters 
obtained from the curve_fit() function, which is stored in sigma.

It then uses the err_ranges() function from the err_ranges module, 
which is using the data_fitting, the model, the optimal parameters 
and the standard deviation to calculate the low and upper error range, 
and assigns it to the variables low and up.

It then creates a new variable x_line which is an array of values 
ranging from the minimum value of the first column of the data_fitting 
to the maximum value of the first column of the data_fitting plus 1.

And creates a new variable y_line which is the result of applying the 
model function to the values of the x_line with the optimal parameters 
obtained from the curve_fit() function. It is important to note that this
 is just creating the values for the line, but it is not plotting it yet.


"""

plt.scatter(x_axis, y_axis,color="blue")

plt.plot(x_line, y_line, '--', color='black')
plt.fill_between(x_line, low, up, alpha=0.7, color='green')
plt.title('China Data Fitting')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()


#country_comparision=gdp_data.iloc[[121, 149,223], 1:]

country_comparision=gdp_country_data.iloc[[121,223],[30,35,
                                            40,45,50,55,60]].fillna("0.0")


country_comparision




#soma=country_comparision.rename(={37:"Algeria",60:"Switzerland" })

country_comparision.index = ['Kenya', 'Sweden']

country_comparision.plot.bar(rot=45,
                    )
                             

plt.title('COUNTRY COMPARISON')
plt.xlabel('Cluster 0 and 1')
plt.ylabel('Value in current US$')
plt.legend( shadow=True)
plt.show()

