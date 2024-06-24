from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import polars as pl

# Load the trained LightGBM model
model = lgb.Booster(model_file="out/models/lgb_model.txt")

# Initialize FastAPI app
app = FastAPI()


# Define the input data schema
class PredictionInput(BaseModel):
    features: dict


# Define the output data schema
class PredictionOutput(BaseModel):
    prediction: float


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input data to a Polars DataFrame
        data = pl.DataFrame([input_data.features])

        # Make prediction
        prediction = model.predict(
            data.to_pandas()
        )  # Convert to pandas DataFrame for LightGBM compatibility

        # Return the prediction
        return PredictionOutput(prediction=prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
