import requests
import json

#from azureml.core import Workspace

#ws = Workspace.from_config()

# URL for the web service, should be similar to:
# 'http://159a4ac6-4925-4238-bd97-dbe0430d3a97.australiaeast.azurecontainer.io/score'

# If the service is authenticated, set the key or token
key = "Qwd1Rhrce8VwE6fHSPHIhDvJ44AVBVJR"

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "Preg": 1,
            "plas": 150,
            "pres": 80,
            "skin": 29,
            "insu": 0,
            "mass": 28.1,
            "pedi": 0.627,
            "age": 25,
            "class": "tested_positive"
          },
           {
            "Preg": 3,
            "plas": 150,
            "pres": 80,
            "skin": 29,
            "insu": 0,
            "mass": 28.1,
            "pedi": 0.627,
            "age": 35,
            "class": "tested_positive"
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


