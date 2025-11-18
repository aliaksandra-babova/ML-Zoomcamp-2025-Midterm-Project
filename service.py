import requests

url = 'http://localhost:9696/predict'  

datapoint = {
    "outlook": "overcast",
    "temperature": "mild",
    "humidity": "normal",
    "wind": "strong"
}

response = requests.post(url, json=datapoint)
result = response.json()

print('response:', result)