import numpy as np 

class Hopfield:

    def __init__(self, patterns):
        self.patterns = patterns
        # cantidad de neuronas en la red = dimensión de un patrón
        self.N = len(patterns[0])
        # calculo W usando la forma que explico Juliana --> poner todos lo patrones en columnas 
        self.weights = np.dot(patterns.T, patterns) / self.N

       

        # numpy.fill_diagonal(a, val, wrap=False)
        np.fill_diagonal(self.weights, 0)

        print("printing weights")
        print(self.weights)

    # actualizar los elementos del vector de estado S(t)
    def update_S(self, current_S):

        # 5x5 . 5x5 --> 5x5
        h = np.dot(self.weights, current_S)

        # h = [ 1 ........ 25]
        h = h.flatten()

        print("gero:")
        print(h)


        print("printing len curren_S")
        print(len(current_S) )


        print("printing range len curren_S")
        print( range(len(current_S)) )


        aux = np.array([])
        for i in range(len(current_S)):
            if h[i] > 0:
                new_value = 1
            if h[i] < 0:
                new_value = -1
            if h[i] == 0:
                new_value = current_S[i]
            aux = np.append(aux, new_value)

        print("printing aux1")
        print(aux)
        return aux



    def train(self, input_pattern, max_iterations):
        iteration = 0
        print("Época: ", iteration)

        #current_S --> 5x5
        current_S = input_pattern
        print(current_S)

        iteration+=1
        print("Época: ", iteration)

        new_S = self.update_S(current_S)
        print(new_S)

        while iteration < max_iterations and not np.array_equiv(current_S, new_S ):   
            current_S = new_S
            iteration += 1
            print("Época: ", iteration)
            new_S = self.update_S(current_S)
            print(new_S)
        return current_S
        
            