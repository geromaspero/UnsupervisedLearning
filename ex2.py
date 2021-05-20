import numpy as np 
from hopfield import Hopfield
from letters import get_letter_bitmap

def ex2():
    def randomize_pattern(letter, probability, conserve_original_letter):
        aux = get_letter_bitmap(letter.upper())
        result = [i for row in aux for i in row]
        for i in range(len(result)):
            if(probability > np.random.uniform(0,1)):
                if conserve_original_letter:
                    if(result[i] == -1):
                        result[i] = 1
                else: 
                    if(result[i] == 1):
                        result[i] = -1
                    else:
                        result[i] = 1
        return result

    def parse(file_name):  
        f = open(file_name, "r")
        aux = f.read()
        result = []
        letter_counter = 0
        for i in range(len(aux)):
            character = aux[i]
            count = 0
            if character != '\n' and character !=',' and character != ' ':
                if character == '#':
                    value = 1
                if character == '.':
                    value = -1
                count += 1
                result = np.append(result, value)
            elif character == ',':
                letter_counter += 1
        return result.reshape(letter_counter, 25)
            

    pattern = parse("patterns.txt")
    hp = Hopfield(pattern)

    #El true o false, indica si se respeta o no la letra
    #al poner false, te cambia cualquier cosa.
    #al poner true, los # de la letra original se mantienen intactos
    test_pattern = randomize_pattern("L", 0.6, True)
    hp.train(test_pattern, 10)



