import json
from ex1a import ex1a
from ex1b import ex1b
from ex2 import ex2

with open('config.json', 'r') as j:
      json_data = json.load(j)
      print(json_data)

if json_data['exercise'] == 1:
    ex1a()
    ex1b()
else:
    ex2()