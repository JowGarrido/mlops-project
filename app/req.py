import requests
from random import randrange

url = "http://localhost:8080/predict"
myobj = {"instances":[["flower_0"]]}

#x = requests.post(url, json=myobj)

# Executando 10 inferências síncronas em série
for _ in range(1):
    # Definindo um id aleatório
    myobj = {"instances":[[f"flower_{randrange(129)}", f"flower_{randrange(129)}"]]}

    # Resultado da predição
    response = requests.post(url, json=myobj).json()
    print(response["predictions"])