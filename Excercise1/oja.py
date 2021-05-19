import random
import numpy as np
import matplotlib.pyplot as plt

class Oja:
    
    def train(self, eta, training_set, epochs):
        input_set = training_set
        inputs_cant = len(input_set.T[0])
        attributes_cant = len(input_set[0])
        weights = np.random.uniform(0, 1, attributes_cant)
        
        for epoch in range(epochs):
            for aux in input_set:
                s = np.inner(aux, weights)
                weights = weights + eta * s * (aux - np.dot(s,weights))

        norma = np.sqrt(np.inner(weights,weights))
        return weights / norma

    def print_results(self, pca1, training_set, countries):
        countries_pca1 = [np.inner(pca1,training_set[i]) for i in range(len(training_set))]
        print("Primera componente principal usando la regla de OJA:")
        print(pca1)
        fig,ax = plt.subplots(1,1)
        bar = ax.bar(countries,countries_pca1)
        ax.set_ylabel('Primera componente principal (PCA1)')
        ax.set_title('PCA1 por país usando OJA')
        ax.set_xticks(range(len(countries)))
        ax.set_xticklabels(countries, rotation=90)
        plt.show()

    
