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
- Original accuracy on validation dataset: 0.96092
- Parameters tuning:
  - eta = 0.01: no improvement
  - max_depth = 5: 0.004 improvement in ROC-AUC
  - min_child_weight = 1: no improvement
- Final testing on full dataset: ROC-AUC: 0.9633681885262746
              

 - **Conclusion**: A simple logistic regression model performs good enough 

## Running locally
Install uastapi, requests and uvicorn:\
`pip install fastapi uvicorn requests`\
Run the app with uvicorn: \
`uvicorn predict:app --host 0.0.0.0 --port 9696 --reload`\
Change the "datapoint inside the service.py file: \
`datapoint = {
    "outlook": "overcast",
    "temperature": "mild",
    "humidity": "normal",
    "wind": "strong"
}`\
Run service.py:\
`python service.py`

## Running Docker
Build the docker image:\
`docker build -t play-prediction .`\
Run it:\
`docker run -it --rm -p 9696:9696 play-prediction`
