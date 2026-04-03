import pickle
import numpy as np
import requests
import sqlite3  
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import os

import requests_cache
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry


app = Flask(__name__)


# ----------------------------------------------------------------------------
# configuration - paths to saved artifacts (adjust as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(BASE_DIR, "model_trained_and_scalars")

MODEL_PATH = os.path.join(folder_path, "model18feb.pkl")
FEATURE_SCALER_PATH = os.path.join(folder_path, "feature_scaler18feb.pkl")
TARGET_SCALER_PATH = os.path.join(folder_path, "target_scaler18feb.pkl")
print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

# ----------------------------------------------------------------------------
# load model and scalers at startup
try:
    print("Loading model from:", MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ MODEL LOAD ERROR:", e)
    model = None

try:
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)
except Exception:
    target_scaler = None

try:
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)
    print("✅ Feature scaler loaded")
except Exception as e:
    print("❌ FEATURE SCALER LOAD ERROR:", e)
    feature_scaler = None


# helper that prepares input data for prediction
# expects JSON payload with a list of 24 hourly records, each itself a list or
# dict of features in the same order used during training.
# minimal validation is performed here; expand in production.

# ---------- helper utilities for backend developers ----------
delhi_holidays_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-14",  # Makar Sankranti / Pongal
    "2026-01-26",  # Republic Day
    "2026-03-04",  # Maha Shivaratri
    "2026-03-21",  # Holi
    "2026-03-31",  # Ramzan Id (Eid-ul-Fitr) *
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-18",  # Good Friday
    "2026-05-01",  # Labour Day
    "2026-05-25",  # Buddha Purnima
    "2026-08-15",  # Independence Day
    "2026-08-28",  # Janmashtami
    "2026-09-17",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-01",  # Diwali
    "2026-11-15",  # Guru Nanak Jayanti
    "2026-12-25"   # Christmas
]

def fetch_weather(startDate,endDate):
    """
    Fetch next 24-hour weather data using Open-Meteo API

    Returns:
        list: 24 hourly weather dictionaries with:
              temp, dewpoint, humidity,
              windspeed, winddir, cloud, precip
    """

    lat = 52.52
    lon = 13.41

    # Setup cached + retry session
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover"
        ],
        "start_date": startDate,
	    "end_date": endDate, 
    }

    try:
        responses = openmeteo.weather_api(url, params=params,verify=False)
    except Exception as e:
        raise Exception(f"Weather API request failed: {e}")

    response = responses[0]
    hourly = response.Hourly()

    # Extract variables (order matters)
    temp = hourly.Variables(0).ValuesAsNumpy()
    humidity = hourly.Variables(1).ValuesAsNumpy()
    dewpoint = hourly.Variables(2).ValuesAsNumpy()
    precip = hourly.Variables(3).ValuesAsNumpy()
    windspeed = hourly.Variables(4).ValuesAsNumpy()
    winddir = hourly.Variables(5).ValuesAsNumpy()
    cloud = hourly.Variables(6).ValuesAsNumpy()

    # Build dataframe (optional but clean)
    hourly_data = pd.DataFrame({
        "temp": temp,
        "humidity": humidity,
        "dewpoint": dewpoint,
        "precip": precip,
        "windspeed": windspeed,
        "winddir": winddir,
        "cloud": cloud
    })

    # ✅ Take only next 24 hours
    hourly_data = hourly_data.head(24)

    # Convert to list of dicts (same as your old function)
    weather_features = hourly_data.to_dict(orient="records")

    if not weather_features:
        raise ValueError("No hourly weather data found")

    return weather_features
def is_delhi_holiday(date_str):
    return date_str in delhi_holidays_2026


@app.route("/hourly-weather", methods=["GET"])
def hourly_weather():
    """
    Test route to check if weather API is working.
    Returns 24-hour weather forecast.
    """

    try:
        weather_data = fetch_weather("2026-03-20","2026-03-20")
        return jsonify(weather_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_holidays(db_path, start_date, end_date): #To be done
    """Get holidays from database.
    
    Input:
        db_path (str): Path to SQLite database
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
    
    Output:
        set: Set of holiday dates as strings (YYYY-MM-DD)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT date FROM holidays WHERE date >= ? AND date <= ?", 
                (start_date, end_date))
    holidays = {r[0] for r in cur.fetchall()}
    conn.close()
    return holidays


def is_weekend(dt): #checked
    """Check if date is weekend (Sat/Sun).
    
    Input:
        dt (str or datetime): Date as ISO string (YYYY-MM-DD) or datetime object
    
    Output:
        bool: True if Saturday or Sunday, False otherwise
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.weekday() >= 5



def preprocess(features):
    arr = np.array(features, dtype=float)

    if arr.ndim != 2 or arr.shape[0] != 24:
        raise ValueError("Input must be 24 rows of features")

    if feature_scaler is not None:
        flat = arr.reshape(-1, arr.shape[1])
        scaled = feature_scaler.transform(flat).reshape(arr.shape)
    else:
        scaled = arr

    return scaled[np.newaxis, ...]


def postprocess(pred):
    """Inverse scale predictions back to original range.
    
    Input:
        pred (np.array): Raw model output, shape (1, 24, 1) or similar
    
    Output:
        list: 24 unscaled predictions (original value range)
    """
    raw = pred.reshape(-1, 1)
    if target_scaler is not None:
        inv = target_scaler.inverse_transform(raw)
    else:
        inv = raw
    return inv.flatten().tolist()


@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint.
    
    Input: (GET request, no body)
    
    Output:
        JSON: {"status": "ok" or "error", "model_loaded": true/false}
    """
    ok = model is not None
    return jsonify({"status": "ok" if ok else "error", "model_loaded": ok})



@app.route("/predict", methods=["GET"])
def predict_auto():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    try:
        # NEW: Get start & end date from UI (epoch format)
        start_epoch = request.args.get("start_date")
        end_epoch = request.args.get("end_date")

        if not start_epoch or not end_epoch:
            return jsonify({"error": "start_date and end_date (epoch) are required"}), 400

        # Convert epoch → datetime
        start_dt = datetime.fromtimestamp(int(start_epoch))
        end_dt = datetime.fromtimestamp(int(end_epoch))

        # Convert to required formats
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        # (Optional: dd-mm-yyyy if needed somewhere)
        formatted_start = start_dt.strftime("%d-%m-%Y")
        formatted_end = end_dt.strftime("%d-%m-%Y")

        # Step 2: Fetch weather data (range)
        weather_data = fetch_weather(start_date, end_date)

        features = []

        for i, w in enumerate(weather_data):
            # IMPORTANT: build datetime sequentially from start date
            dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=i)

            hour = dt.hour
            day = dt.day
            month = dt.month
            year = dt.year
            dayofweek = dt.weekday()

            # Step 4: Holiday & weekend
            is_holiday = int(is_delhi_holiday(start_date))
            weekend_flag = int(is_weekend(dt))

            # Step 5: Combine ALL features in correct order
            row = [
                hour,
                day,
                month,
                year,
                dayofweek,
                w["temp"],
                w["dewpoint"],
                w["humidity"],
                w["windspeed"],
                w["winddir"],
                w["cloud"],
                w["precip"],
                is_holiday,
                weekend_flag
            ]

            features.append(row)

        # Step 6: Preprocess
        inp = preprocess(features)

        # Step 7: Predict
        pred_raw = model.predict(inp)

        # Step 8: Postprocess
        result = postprocess(pred_raw)

        # Safety: ensure alignment
        if len(result) != len(features):
            result = result[:len(features)]

        # Start from midnight of start_date
        base_time = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        response_data = []

        for i, value in enumerate(result):
            dt = base_time + timedelta(hours=i)
            unix_time = int(dt.timestamp())

            response_data.append({
                "time": unix_time,
                "value": float(value)
            })

        return jsonify({
            "status": "success",
            "predictions": response_data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # optional debug flag; in production use gunicorn or similar
    app.run(host="0.0.0.0", port=5000, debug=True)


