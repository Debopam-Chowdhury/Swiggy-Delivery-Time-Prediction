from flask import Flask, request, render_template, jsonify
from sklearn.pipeline import Pipeline
import pandas as pd
import mlflow
import json
import joblib
from sklearn import set_config
from mlflow import MlflowClient
import os

# Optional: if using Dagshub
import dagshub
dagshub.init(repo_owner='Debopam-Chowdhury', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)

# Set the output as pandas
set_config(transform_output='pandas')

# Set MLflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Debopam-Chowdhury/Swiggy-Delivery-Time-Prediction.mlflow")

# import dagshub
# dagshub.init(
#     repo_owner='Debopam-Chowdhury',
#     repo_name='Swiggy-Delivery-Time-Prediction',
#     mlflow=True,
#     token=os.environ.get("DAGSHUB_TOKEN"),
#     user=os.environ.get("DAGSHUB_USERNAME")
# )

# mlflow.set_tracking_uri("https://dagshub.com/Debopam-Chowdhury/Swiggy-Delivery-Time-Prediction.mlflow")

# ========== Load model metadata & transformer ==========
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# ========== Feature Columns ==========
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle',
                    "festival", "city_type", "is_weekend", "order_time_of_day"]

ordinal_cat_cols = ["traffic", "distance_type"]

# MLflow Client
client = MlflowClient()

# Load model name and stage
model_name = load_model_information("run_information.json")['model_name']
stage = "Production"

# Load model path and model
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Load preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# Build pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', model)
])

# ========== Flask App ==========
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    pred_data = pd.DataFrame({
        'age': data['age'],
        'ratings': data['ratings'],
        'weather': data['weather'],
        'traffic': data['traffic'],
        'vehicle_condition': data['vehicle_condition'],
        'type_of_order': data['type_of_order'],
        'type_of_vehicle': data['type_of_vehicle'],
        'multiple_deliveries': data['multiple_deliveries'],
        'festival': data['festival'],
        'city_type': data['city_type'],
        'is_weekend': int(data['is_weekend']),
        'pickup_time_minutes': data['pickup_time_minutes'],
        'order_time_of_day': data['order_time_of_day'],
        'distance': data['distance'],
        'distance_type': data['distance_type']
    }, index=[0])

    predictions = model_pipe.predict(pred_data)[0]
    return jsonify({"predicted_delivery_time": round(predictions, 2)})

if __name__ == "__main__":
    app.run(debug=True)
