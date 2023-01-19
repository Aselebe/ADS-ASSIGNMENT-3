# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:54:42 2023

@author: haselebe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
import err_ranges as err
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



Regenerate response
"""

def read_data(url):
    data = pd.read_excel(url, skiprows=3)
    
    data = data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], 
                     axis=1)

    #this extracts a dataframe with countries as column
    df_country = data
    
    #this section extract a dataframe with year as columns
    df_years = data.transpose()

    #removed the original headers after a transpose and dropped the row
    #used as a header
    df_years = df_years.rename(columns=df_years.iloc[0])
    df_years = df_years.drop(index=df_years.index[0], axis=0)
    df_years['Year'] = df_years.index
    #df2 = df2.rename(columns={"index":"Year"})
    return df_country, df_years

gdp_country_data, gdp_year_data = read_data(url)

#extract the required data for the clustering
gdp_data = gdp_country_data.loc[gdp_country_data.index, ['Country Name', 
                                                         year1, year2]].dropna()

gdp_data

#convert the datafram to an array
x = gdp_data[[year1, year2]].dropna().values
x

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

sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,
                    random_state=0)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
    
    
plt.plot(range(1,11), sse, marker="o", color="red")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, 
                random_state=0)
y_kmeans = kmeans.fit_predict(x)

y_kmeans

#this creates a new dataframe with the labels for each country
gdp_data['label'] = y_kmeans
df_label = gdp_data.loc[gdp_data['label'] == 0]
df_label.head(20)

y = kmeans.cluster_centers_
y

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

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'purple',
            label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'orange',
            label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green',
            label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 10, c = 'red', label = 'Centroids')
plt.legend()
plt.show()

def model(x, a, b, c, d):
    '''
    docstring
    
    '''
    return a*x**3 + b*x**2 + c*x + d

"""
This code is using the scipy.optimize module to fit a curve to a set of data.
 The curve_fit() function is being used to fit the model function to the data 
 in x_axis and y_axis. The function returns two variables: popt contains the 
 optimal values for the parameters of the model function, and covar contains
 the covariance of the parameters.


"""
fitting = gdp_year_data[['Year', 'China']].apply(pd.to_numeric,errors='coerce')

fitting.plot("Year", "China", kind="scatter")

data_fitting = fitting.dropna().values

x_axis = data_fitting[:,0]
y_axis = data_fitting[:,1]

"""
It then uses tuple unpacking to assign the optimal values of the parameters
 of the model function (a, b, c, d) to the variables a, b, c and d 
 respectively. It is important to note that the number of variables 
 should match the number of parameters in the model function.
"""
popt, _ = opt.curve_fit(model, x_axis, y_axis)
param, covar = opt.curve_fit(model, x_axis, y_axis)
a, b, c, d = popt

import numpy as np

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
sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(data_fitting, model, popt, sigma)

plt.scatter(x_axis, y_axis)

plt.scatter(x_axis, y_axis)

x_line = np.arange(min(data_fitting[:,0]), max(data_fitting[:,0])+1, 1)
y_line = model(x_line, a, b, c, d)


"""
This code is plotting the data, the fitted line, and the error ranges
 on the same graph using matplotlib.

It first plots the scatter plot of the original data using the x_axis and
 y_axis variables, then it plots the line of best fit using the x_line and
 y_line variables, with a dashed black line style.

Then it uses the fill_between() function to fill the area between the low 
and upper error ranges, which are calculated using the err_ranges() function 
earlier. The color of the fill is set to green and the transparency
 is set to 0.7.

It then sets the title of the plot to "Data Fitting", labels the x-axis as
 "Year" and the y-axis as "". Finally, it uses the show() function to display
 the plot.
 
It uses plt.show() twice, the second one is redundant and can be removed.




Regenerate response

"""

plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line, '--', color='black')
plt.fill_between(data_fitting, low, up, alpha=0.7, color='green')
plt.title('Data Fitting')
plt.xlabel('Year')
plt.ylabel('')
plt.show()
plt.show()

