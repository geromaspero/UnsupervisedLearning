import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from oja import Oja
import matplotlib.pyplot as plt
import json

def load_csv(file_path): 
    
    parsed_file = pd.read_csv(file_path)
    
    countries_list = parsed_file.loc[ : ,"Country"]
    countries_list_values = countries_list.values
    
    properties = parsed_file.iloc[ : , 1:8 ]
    properties_values = properties.values
    scaler = StandardScaler()
    scaled_properties_values = scaler.fit_transform(properties_values)
    
    return countries_list_values, scaled_properties_values

def print_libarary_pca1(pca1, training_set, countries):
    countries_pca1 = [np.inner(pca1,training_set[i]) for i in range(len(training_set))]
    print("Primera componente principal usando la librería Sklearn:")
    print(pca1)
    fig,ax = plt.subplots(1,1)
    bar = ax.bar(countries,countries_pca1)
    ax.set_ylabel('Primera componente principal (PCA1)')
    ax.set_title('PCA1 por país usando la librería Sklearn')
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=90)
    plt.show()

(countries, training_set) = load_csv('europe.csv')


df = pd.read_csv('europe.csv')
X_cols = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
std_df = StandardScaler().fit_transform(df[X_cols])
pca = PCA(n_components=7)
lib_components = pca.fit_transform(std_df)[:,0]

print_libarary_pca1(pca.components_[0], training_set, countries)


oja_component = Oja()
with open('config.json', 'r') as j:
    json_data = json.load(j)
    print(json_data)

pca1 = oja_component.train(json_data['eta'], training_set, json_data['epochs'])
if pca.components_[0][0] * pca1[0] < 0:
    pca1 = pca1 * -1
oja_component.print_results(pca1, training_set, countries)


