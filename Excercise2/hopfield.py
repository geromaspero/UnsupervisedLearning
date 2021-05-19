import numpy as np 
import sys

class Hopfield:

    def __init__(self, patterns):
        self.patterns = patterns
        # cantidad de neuronas en la red = dimensión de un patrón
        self.N = len(patterns[0])
        # calculo W usando la forma que explico Juliana --> poner todos lo patrones en columnas 
        self.weights = np.dot(patterns.T, patterns) / self.N

       

        # numpy.fill_diagonal(a, val, wrap=False)
        np.fill_diagonal(self.weights, 0)

        

    # actualizar los elementos del vector de estado S(t)
    def update_S(self, current_S):

        # 5x5 . 5x5 --> 5x5
        h = np.dot(self.weights, current_S)

        # h = [ 1 ........ 25]
       # h = h.flatten()



        aux = np.array([])
        for i in range(len(current_S)):
            if h[i] > 0:
                new_value = 1
            if h[i] < 0:
                new_value = -1
            if h[i] == 0:
                new_value = current_S[i]
            aux = np.append(aux, new_value)

        
        return aux


    def print_nice(self,pattern):
        for i in range(5):
            for j in range(5):
                aux = pattern[i * 5 + j]
                if aux == 1:
                    character = "#"
                else:
                    character = "."
                sys.stdout.write(character)
            print("")
        return


    def train(self, input_pattern, max_iterations):

      

        

        iteration = 0
        print("Época: ", iteration)

        #current_S --> 5x5
        current_S = input_pattern
        self.print_nice(current_S)

        iteration+=1
        print("Época: ", iteration)

        new_S = self.update_S(current_S)
        self.print_nice(new_S)

        while (not np.array_equal(current_S, new_S)) and iteration <= max_iterations:   
            current_S = new_S
            iteration += 1
            print("Época: ", iteration)
            new_S = self.update_S(current_S)
            self.print_nice(new_S)
        
        return new_S

    