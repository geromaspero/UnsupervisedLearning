letters_bitmap = {
    "A": [[
        -1,-1,1,-1,-1],[
        -1,1,-1,1,-1],[
        -1,1,1,1,-1],[
        1,1,1,1,1],[
        1,-1,-1,-1,1]],
    "B": [[
        1,1,1,1,-1],[
        1,-1,-1,-1,1],[
        1,1,1,1,-1],[
        1,-1,-1,-1,1],[
        1,1,1,1,-1]],
    "C": [[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1]],
    "D": [[
        1,1,1,1,-1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,1,1,1,-1]],
    "E": [[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1]],
    "F": [[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1]],
    "G": [[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,-1,1,1,1],[
        1,-1,-1,-1,1],[
        1,1,1,1,1]],
    "H": [[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,1,1,1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1]],
    "I": [[
        1,1,1,1,1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1],[
        1,1,1,1,1]],
    "J": [[
        1,1,1,1,1],[
        -1,-1,-1,1,-1],[
        -1,-1,-1,1,-1],[
        1,-1,-1,1,-1],[
        1,1,1,-1,-1]],
    "K": [[
        1, -1, -1, -1, 1,],[
        1, -1, -1, 1, -1],[
        1, 1, 1, -1, -1],[
        1, -1, -1, 1, -1],[
        1, -1, -1, -1, 1]],
    "L": [[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1]],
    "M": [[
        1,-1,-1,-1,1],[
        1,1,-1,1,1],[
        1,-1,1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1]],
    "N": [[
        1, -1, -1, -1, 1],[
        1, 1, -1, -1, 1],[
        1, -1, 1, -1, 1],[
        1, -1, -1, 1, 1],[
        1, -1, -1, -1, 1]],
    "O": [[
        1,1,1,1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,1,1,1,1]],
    "P": [[
        1,1,1,1,-1],[
        1,-1,-1,-1,1],[
        1,1,1,1,-1],[
        1,-1,-1,-1,-1],[
        1,-1,-1,-1,-1]],
    "Q": [[
        -1,1,1,-1,-1],[
        1,-1,-1,1,-1],[
        1,-1,-1,1,-1],[
        1,-1,-1,1,-1],[
        1,1,1,-1,1]],
    "R": [[
        1,1,1,1,-1],[
        1,-1,-1,-1,1],[
        1,1,1,1,-1],[
        1,-1,1,-1,-1],[
        1,-1,-1,1,-1]],
    "S": [[
        1,1,1,1,1],[
        1,-1,-1,-1,-1],[
        1,1,1,1,1],[
        -1,-1,-1,-1,1],[
        1,1,1,1,1]],
    "T": [[
        1,1,1,1,1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1]],
    "U": [[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,1,1,1,1]],
    "V": [[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        1,-1,-1,-1,1],[
        -1,1,-1,1,-1],[
        -1,-1,1,-1,-1]],
    "W": [[
        1,-1,1,-1,1],[
        1,-1,1,-1,1],[
        1,-1,1,-1,1],[
        1,-1,1,-1,1],[
        -1,1,-1,1,-1]],
    "X": [[
        1,-1,-1,-1,1],[
        -1,1,-1,1,-1],[
        -1,-1,1,-1,-1],[
        -1,1,-1,1,-1],[
        1,-1,-1,-1,1]],
    "Y": [[
        1,-1,-1,-1,1],[
        -1,1,-1,1,-1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1],[
        -1,-1,1,-1,-1]],
    "Z": [[
        1,1,1,1,1],[
        -1,-1,-1,1,-1],[
        -1,-1,1,-1,-1],[
        -1,1,-1,-1,-1],[
        1,1,1,1,1]],
}

def get_letter_bitmap(letter):
    return letters_bitmap.get(letter)