import numpy as np 
from hopfield import Hopfield

def parse(file_name):  

    f = open(file_name, "r")
    aux = f.read()

    result = np.array([])

    for i in range(len(aux)):
        character = aux[i]

        if character != '\n':
            if character == '#':
                value = 1
            if character == '.':
                value = -1
            result = np.append(result, value)

    return result.reshape(5,5)
        

pattern = parse("patterns.txt")
hp = Hopfield(pattern)

test = parse("test.txt")
print(hp.train(test, 10))



