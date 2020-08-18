import json
import requests
import pandas as pd
import numpy as np

port = 1237

def req(liste):

    col_names=["Pclass","Sex","Age"]
    test = pd.DataFrame(list(liste),columns=col_names)

    input_json = test.to_json(orient='split')
    input_json = json.loads(input_json)
    endpoint = "http://localhost:{}/invocations".format(port)
    headers = {"Content-Type": "application/json"}
    #; format=pandas-split
    print(input_json)
    pred = requests.post(endpoint,json=input_json,headers=headers)

    return pred.text[1] #pred est sous la forme '[x]' donc on choisit le 2 eme elem


# JSON POST REQUEST :
# {
# "columns": ["Pclass", "Sex", "Age"],
# "index": [0],
# "data": [[1, 2, 33]]
# }
