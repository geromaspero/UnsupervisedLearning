# -*- coding: utf-8 -*-
"""1.a.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DuE4BVe-dTVeN71eRpMFSSAh4ltrCqQ3

Importing Libraries
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

def ex1a():
  with open('config.json', 'r') as j:
      json_data = json.load(j)
      print(json_data)


  """Opening 'europe.csv'"""
  """Putting file contents in a Data Frame"""
  df = pd.read_csv(json_data['file_to_read'])
  print(df)

  """We need to separate countries names from all other properties"""
  columns_names_list = df.columns.tolist()
  columns_names_list.pop(0)
  countries = np.array(df.loc[ : ,"Country"].values)
  properties = np.array(df.iloc[ : , 1:8 ].values)
  print(columns_names_list)
  print(countries)
  print(properties)

  """We normalize the data"""
  mean, std = np.mean(properties, axis=0), np.std(properties, axis=0)
  print('mean: ', mean)
  print('std deviation: ', std)
  properties_norm = (properties - mean)/std
  print(properties_norm)

  """We create Neuron class"""
  class Neuron(object):
    def __init__(self, _attributes = None):
      self._attributes = _attributes

    def __str__(self):
      return str(self._attributes)

    def __repr__(self):
      return str(self._attributes)

    def __eq__(self, other):
      return other._attributes == self._attributes

    def get_attributes(self):
      return self._attributes
    
    def set_attributes(self, attributes):
      self._attributes = attributes
    
    def del_attributes(self):
      del self.attributes
    
    attributes = property(get_attributes, set_attributes, del_attributes)

  """Ex. 1.a

  We create the main variables (epochs, neurons, attributes_Q, eta0, eta_arr, W0, W_arr, radius)
  """
  epochs = json_data['epochs']
  neurons = json_data['neurons']
  attributes_Q = json_data['attributes_Q']
  eta0 = np.random.rand()
  eta_arr = np.array([eta0])
  W0 = np.full((neurons,neurons), Neuron())
  for i in range(neurons):
    for j in range(neurons):
      W0[i][j] = Neuron(properties_norm[i*neurons + j])
  W_arr = np.array([W0])
  radius = json_data['radius']

  print('epochs: ' ,epochs)
  print('neurons: ', neurons)
  print('attributes_Q: ', attributes_Q)
  print('eta0: ', eta0)
  print('eta_arr: ', eta_arr)
  print('Wo: ', W0)
  print('Warr: ', W_arr)
  print('radius: ', radius)

  """Function to select best neuron for a given row using Euclidean Distance - selectBestW(W, row)"""
  def selectBestW(row, t):
    W_temp = W_arr[t]
    min = np.linalg.norm(row - W_temp[0][0].attributes)
    bestX = 0
    bestY = 0
    for i in range(neurons):
      for j in range(neurons):
        if min > np.linalg.norm(row - W_temp[i][j].attributes):
          bestX = i
          bestY = j
          min = np.linalg.norm(row - W_temp[i][j].attributes)
    return bestX, bestY

  """Function to update W with adding delta W of the given equation
  eta(t) = 1/t if t != 0
  """
  def updateW(bestWX, bestWY, t, row):
    W_temp = np.copy(W_arr[t])
    for i in range(neurons):
      for j in range(neurons):
        if np.sqrt((bestWX-i)**2 + (bestWY-j)**2) <= radius:
          W_temp[i][j].attributes = W_temp[i][j].attributes + eta_arr[t]*(row - W_temp[i][j].attributes)
    return W_temp

  """1. We calculate the minor euclidean distance from a random row of the properties each neuron
  2. We update W
  3. We start over
  """
  for t in range(epochs):
    for i in range(len(properties)):
      r = np.random.randint(len(properties_norm))
      row = properties_norm[i]
      bestWX, bestWY = selectBestW(row, t)
      W = updateW(bestWX, bestWY, t, row)
      W_arr = np.append(W_arr, [W],0)
    eta_arr = np.append(eta_arr, 1/(t+1))

  """We print each neuron"""
  for i in range(len(W)):
    for j in range(len(W[i])):
      print('W[{}][{}]: {}'.format(i,j,W[i][j].attributes))

  """We create a Matrix with each Neuron and the countries that are in that neuron"""
  chosen_neurons = []
  for c in range(len(properties_norm)):
    chosen_neuron = W[0][0]
    for i in range(len(W)):
      for j in range(len(W[i])):
        diff = np.sum(np.abs(W[i][j].attributes - properties_norm[c]))
        chosen_diff = np.sum(np.abs(chosen_neuron.attributes - properties_norm[c]))
        if diff < chosen_diff:
          chosen_neuron = W[i][j]
    chosen_neurons.append(chosen_neuron)

  m = np.zeros((len(W),len(W[0])))
  counts = [ [ [] for i in range(len(W)) ] for j in range(len(W)) ]
  for c in range(len(properties_norm)):
    for i in range(len(W)):
      for j in range(len(W[i])):
          if all(chosen_neurons[c].attributes == W[i][j].attributes):
            counts[i][j].append(countries[c])
            m[i][j] += 1
  print(m)
  print(counts)

  """U Matrix"""
  u_matrix = np.zeros((len(W),(len(W[0]))))
  for i in range(len(W)):
    for j in range(len(W[i])):
      average_euclidean_distance = 0
      neighbors = 0
      if i != 0:
        average_euclidean_distance += np.linalg.norm(W[i-1][j].attributes - W[i][j].attributes)
        neighbors += 1
      if i != len(W) - 1:
        average_euclidean_distance += np.linalg.norm(W[i+1][j].attributes - W[i][j].attributes)
        neighbors += 1
      if j != 0:
        average_euclidean_distance += np.linalg.norm(W[i][j-1].attributes - W[i][j].attributes)
        neighbors += 1
      if j != len(W[i]) - 1:
        average_euclidean_distance += np.linalg.norm(W[i][j+1].attributes - W[i][j].attributes)
        neighbors += 1
      average_euclidean_distance /= neighbors
      u_matrix[i][j] = average_euclidean_distance
  print(u_matrix)
  sns.heatmap(u_matrix, cmap=sns.color_palette("YlOrBr", as_cmap=True))

  """We do a Heatmap"""
  fig, ax = plt.subplots()
  sns.heatmap(m, cmap=sns.color_palette("YlOrBr", as_cmap=True), vmin=0, vmax=15)
  plt.show()

  """We show the data separated in groups in the biplot"""
  #------------------ data parsing ------------------#
  parsed_file = df
  #------------------ data parsing ------------------#
  columns_names_list = parsed_file.columns.tolist()
  columns_names_list.pop(0)
  #------------------ list of countries ------------------# 
  countries_list = parsed_file.loc[ : ,"Country"]

  countries_list_values = countries_list.values 
  #------------------ list of countries ------------------#
  #------------------ countries' properties ------------------#
  properties = parsed_file.iloc[ : , 1:8 ]
  properties_values = properties.values
  #------------------ countries' properties ------------------#
  #------------------ Standarization of data ------------------#
  # define standard scaler
  scaler = StandardScaler()

  # transform data
  scaled_properties_values = scaler.fit_transform(properties_values)
  #------------------ Standarization of data ------------------#
  #------------------ covariance matrix ------------------#
  covM = pd.DataFrame(scaled_properties_values)
  covariance_matrix = pd.DataFrame.cov(covM)
  #------------------ covariance matrix ------------------#
  #------------------ eigenvalues (w) & eigenvectors (v) ------------------#
  w, v = np.linalg.eig(covariance_matrix)
  w = sorted(w, reverse=True) 
  #------------------ PCA Calculation ------------------#
  pca = PCA(n_components=7).fit(scaled_properties_values)
  #------------------ PCA Calculation ------------------#
  pca2 = PCA(n_components=7)
  components = pca2.fit_transform(scaled_properties_values)
  loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)

  colored_countries = [ "" for i in range(len(countries)) ]
  for i in range(len(counts)):
    for j in range(len(counts[i])):
      for k in range(len(counts[i][j])):
        for c in range(len(countries)):
          if counts[i][j][k] == countries[c]:
            colored_countries[c] = 2*(i*len(counts) + j)


  fig = px.scatter(components, x=0, y=1, text=countries, color=colored_countries)
  fig.update_layout(font=dict(size=15))
  fig.update_traces(textposition='bottom center')
  #we add the vectors to the figure
  for component in components:
    fig.add_annotation()
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