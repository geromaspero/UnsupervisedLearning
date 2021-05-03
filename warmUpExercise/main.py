import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import plotly.express as px


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




columns_names_list = parsed_file.columns.tolist()
columns_names_list.pop(0)

# print(columns_names_list)
# ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']


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



#------------------ covariance matrix ------------------#

df = pd.DataFrame(scaled_properties_values)
covariance_matrix = pd.DataFrame.cov(df)

#print(covariance_matrix)


# Alternate way to find covariance matrix
#covariance_matrix2 = np.cov(scaled_properties_values.T) 
#print(covariance_matrix2)

#------------------ covariance matrix ------------------#



#------------------ eigenvalues (w) & eigenvectors (v) ------------------#

w, v = np.linalg.eig(covariance_matrix)

#sort in descending order
w = sorted(w, reverse=True) 

#print(w)
# [3.3466903338935152, 1.2310909419122338, 1.1025679615273636, 0.798887683362183, 0.47480597493684956, 0.17492107429275486, 0.1302952893343626]


#------------------ eigenvalues (w) & eigenvectors (v) ------------------#






#------------------ PCA Calculation ------------------#

pca = PCA(n_components=7).fit(scaled_properties_values)

# pca.explained_variance_ratio_: Percentage of variance explained by each of the selected components.

# print(pca.explained_variance_ratio_)
# [0.46102367 0.16958906 0.15188436 0.11005085 0.06540695 0.02409627 0.01794884]


# print(np.cumsum(pca.explained_variance_ratio_))
# [0.46102367 0.63061273 0.78249709 0.89254794 0.95795489 0.98205116 1.00]

#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.show()


#print charges! 
#print(pca.components_)
#[
# [ 1.24873902e-01 -5.00505858e-01  4.06518155e-01 -4.82873325e-01  1.88111616e-01 -4.75703554e-01  2.71655820e-01] --> autovector A
# [-1.72872202e-01 -1.30139553e-01 -3.69657243e-01  2.65247797e-01  6.58266888e-01  8.26219831e-02  5.53203705e-01] --> autovector B
# ...... ]


#EJERCICIO: INTERPRETAR LA PRIMERA COMPONENTE
#Si la componente principal del dataset me da negativo (ex: -2), el autovector A, va a tener mas de de las componentes 
# negativas (mas de "-5.00505858e-01", de "-4.82873325e-01" y de "-4.75703554e-01"


#------------------ PCA Calculation ------------------#



pca2 = PCA(n_components=7)

components = pca2.fit_transform(scaled_properties_values)


loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)


#dots ploted in the figure
# Value of 'x' is not the name of a column in 'data_frame'. Expected one of [0, 1, 2, 3, 4, 5, 6]
# Value of 'y' is not the name of a column in 'data_frame'. Expected one of [0, 1, 2, 3, 4, 5, 6]
fig = px.scatter(components, x=0, y=1, color=countries_list_values)


#we add the vectors to the figure
for i, feature in enumerate(columns_names_list):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )

fig.show()


