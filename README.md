# ML Zoomcamp 2025 Midterm Project: Tennis game likelihood 
The idea of this project is to use historical meterological data to predict the likelihood of tennis players actually playing on a given day. The model can be used as part of an algorithm to predict tennis courts occupacy rate on a given day and proactively manage it by sensding personalised messages encouraging players to come and play during gloomy but dry days that could otherwise put the players off and leave the courts empty.  

# Problem statement
Tennis clubs may experience lower courts occupacy rates during gloomy days, because the players may prefer to avoid risking booking a court and not being able to play due to rain. However, if the weather forecast does not actually predict any rain, the courts can normally be used without any problem. 

## Objective
This project aims to develop a ML model predicting the likelihood of a tennis player to play on a certain day based on the weather forecast. 

## Possible Use Cases
1. Predicting future courts availability or occupacy based on the weather forecast.
2. Preventing possible revelue loss by sending targetted messages to the players when the likelihood of playing is low, but there's no rain forecasted for the day.
3. Analysing past data to understand potential revenue losses due to weather conditions when planning a new tennis club.
   
# Dataset description
The dataset captures how weather conditions🌤️ affect the decision to play tennis.\
The dataset on Kaggle: ['/kaggle/input/play-tennis-dataset-weather-based-classifier/play_tennis_dataset.csv'](https://www.kaggle.com/datasets/milapgohil/play-tennis-dataset-weather-based-classifier)\
The CSV to download: https://raw.githubusercontent.com/aliaksandra-babova/ML-Zoomcamp-2025-Midterm-Project/refs/heads/main/play_tennis_dataset.csv \
Total Records: 6666 rows

# EDA summary
1. Overall distribution:\
`61% 'Yes' / 38% 'No' ('churn')`
2. Feature importance:\
`Temperature:    0.121530`\
`Outlook:        0.119611`\
`Humidity:       0.040637`\
`Wind:           0.033610`

# Modeling approach & metrics
1. Logistic regression
- Original accuracy on validation dataset: 0.900225056264066
- Parameters tuning:
  - C = 1: no improvement
- Final testing:\
`Accuracy: 0.9025487256371814`\
`ROC-AUC: 0.9634685976350151`
 - Cross-validation on 5 folds:
   - Max deviation: 0.002
 - **Conclusion**: acceptable (**ROC-AUC: 0.963**) and stable results with a simple logistic regression model

2. Random forest
- Original accuracy on validation dataset: 0.961
- Parameters tuning:
  - max_depth = 5: 0.002 improvement in ROC-AUC
  - min_samples_leaf = 10: 0.002 improvement in ROC-AUC
  - n_estimators=35: optimal value
- Final testing:
`Random Forest accuracy: 0.8455772113943029`\
`Random Forest ROC-AUC: 0.96403836094973`
 - **Conclusion**: slightly better accuracy (**ROC-AUC: 0.964**) than a simple logistic regression model

3. RGBoost
- Original accuracy on validation dataset: 0.963
- Parameters tuning:
  - eta = 0.05: 0.001 improvement in ROC-AUC
  - max_depth = 6: no improvement
  - min_child_weight = 1: no improvement
- Final testing: ROC-AUC: 0.9644294894547085
              

 - **Conclusion**: A simple logistic regression model performs good enough 

# Testing the predictions
## Running locally
1. Clone the repo:
```
git clone https://github.com/aliaksandra-babova/ML-Zoomcamp-2025-Midterm-Project
```
2. Install uv:
```
pip install uv
```
3. Switch directory:
```
cd ML-Zoomcamp-2025-Midterm-Project
```
4. Install the project's dependencies:
```
uv sync
```
5. Run the app with uvicorn inside the virtual environment: 
```
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```
6. Send a request via the api docs (http://localhost:9696/docs) or with this curl:
```
curl -X 'POST' 'http://localhost:9696/predict' 
   -H 'accept: application/json' 
   -H 'Content-Type: application/json' 
   -d '{
    "outlook": "overcast",
    "temperature": "mild",
    "humidity": "normal",
    "wind": "strong"
   }'
```
7. You can also run the service.py script:
```
python service.py
```

## Running Docker
1. Switch directory:
```
cd ML-Zoomcamp-2025-Midterm-Project
```
2. Build the docker image:
```
docker build -t play-prediction .
```
3. Run it:
```
docker run -it --rm -p 9696:9696 play-prediction
```
4. Send a request via the api docs (http://localhost:9696/docs) or with this curl: 
```
curl -X 'POST' 'http://localhost:9696/predict' 
  -H 'accept: application/json' 
  -H 'Content-Type: application/json' 
  -d '{
    "outlook": "overcast",
    "temperature": "mild",
    "humidity": "normal",
    "wind": "strong"
   }'
```
5. You can also run the service.py script:
```
python service.py
```

## Test online
1. Open https://winter-violet-8319.fly.dev/docs to test the endpoint sending:
```
{
    "outlook": "overcast",
    "temperature": "mild",
    "humidity": "normal",
    "wind": "strong"
   }
```
2. Run the service.py script:
```
python service.py
```
# Next steps
1. I'm not sure about the data quality as I understand it is synthetic. Testing with some reasonable combination of conditions some predictions didn't make sense. So getting a real dataset for solving real problems is important.
2. I didn't implement data validation for the input, it'd be nice to have.
3. API documentation could be improved as well (goes with the previous point), all the features are categorical with very few options.
4. With a real dataset I'd some feature engineering exploring feature dependencies and taking into account real life implications (ex. even moderate wind in combination with cold temperature and high humidity could prevent players from playing). 
