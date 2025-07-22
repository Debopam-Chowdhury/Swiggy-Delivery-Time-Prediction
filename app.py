from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
# import dagshub
import mlflow.client

import dagshub
dagshub.init(repo_owner='Debopam-Chowdhury', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Debopam-Chowdhury/Swiggy-Delivery-Time-Prediction.mlflow")



class Data(BaseModel):  
    age: float
    ratings: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: float
    festival: str
    city_type: str
    is_weekend: int
    pickup_time_minutes: float
    order_time_of_day: str
    distance: float
    distance_type: str

    
    
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer



# columns to preprocess in data
num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "Production"

# get the latest model version
# latest_model_ver = client.get_latest_versions(name=model_name,stages=[stage])
# print(f"Latest model in production is version {latest_model_ver[0].version}")

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

# create the app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# create the home endpoint
# @app.get(path="/")
# def home():
#     return "Welcome to the Swiggy Food Delivery Time Prediction App"

@app.get("/", response_class=None)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'age': data.age,
        'ratings': data.ratings,
        'weather': data.weather,
        'traffic': data.traffic,
        'vehicle_condition': data.vehicle_condition,
        'type_of_order': data.type_of_order,
        'type_of_vehicle': data.type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'festival': data.festival,
        'city_type': data.city_type,
        'is_weekend': int(data.is_weekend),
        'pickup_time_minutes': data.pickup_time_minutes,
        'order_time_of_day': data.order_time_of_day,
        'distance': data.distance,
        'distance_type': data.distance_type
        },index=[0]
    )
    # clean the raw input data
    # cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(pred_data)[0]

    return {"predicted_delivery_time": round(predictions, 2)}
   
# @app.get("/", response_class=None)
# def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
   
if __name__ == "__main__":
    uvicorn.run(app="app:app")