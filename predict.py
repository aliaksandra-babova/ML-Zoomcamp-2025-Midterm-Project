import pickle

import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="play-prediction")


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(datapoint):
    result = pipeline.predict_proba(datapoint)[0, 1]
    return float(result)

@app.post("/predict")
def predict(datapoint: Dict[str, Any]):
    prob = predict_single(datapoint)

    return {
        "Probability of playing": prob,
        "Will play tennis today?": bool(prob >=0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
