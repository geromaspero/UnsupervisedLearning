import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



#------------------ data parsing ------------------#

parsed_file = pd.read_csv('europe.csv')

#This is how the parsed date looks like
#            Country    Area    GDP  ...  Military  Pop.growth  Unemployment
# 0          Austria   83871  41600  ...      0.80        0.03           4.2
# .
# .
# .

# the parsed data has the following shape: [28 rows x 8 columns]

#------------------ data parsing ------------------#



#------------------ list of countries ------------------#

#List of countries 
countries_list = parsed_file.loc[ : ,"Country"]

#List of countries: Array format 
countries_list_values = countries_list.values 

#print(countries_list_values)

# ['Austria' 'Belgium' 'Bulgaria' 'Croatia' 'Czech Republic' 'Denmark'
#  'Estonia' 'Finland' 'Germany' 'Greece' 'Hungary' 'Iceland' 'Ireland'
#  'Italy' 'Latvia' 'Lithuania' 'Luxembourg' 'Netherlands' 'Norway' 'Poland'
#  'Portugal' 'Slovakia' 'Slovenia' 'Spain' 'Sweden' 'Switzerland' 'Ukraine'
#  'United Kingdom']

#------------------ list of countries ------------------#




#------------------ countries' properties ------------------#

properties = parsed_file.iloc[ : , 1:8 ]

#Array format
properties_values = properties.values


#print(properties_values)

#[
# [ 8.38710e+04  4.16000e+04  3.50000e+00  7.99100e+01  8.00000e-01  3.00000e-02  4.20000e+00]
# [ 3.05280e+04  3.78000e+04  3.50000e+00  7.96500e+01  1.30000e+00  6.00000e-02  7.20000e+00]
# ....... ]

#------------------ countries' properties ------------------#



#------------------ Standarization of data ------------------#

# Standardization scales each input variable separately by subtracting
# the mean (called centering) and dividing by the standard deviation 
# to shift the distribution to have a mean of zero and a standard deviation of one.
 

# define standard scaler
scaler = StandardScaler()

# transform data
scaled_properties_values = scaler.fit_transform(properties_values)


#print(scaled_properties_values)

#[
# [-0.50783522  0.68390042  0.11444681  0.57077826 -1.0243472  -0.17678894  -1.24552737]
# [-0.83598724  0.41706139  0.11444681  0.48775597 -0.38895239 -0.11592718  -0.59244186]
# ....... ]


#------------------ Standarization of data ------------------#





