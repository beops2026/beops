import pickle
import numpy as np
import requests
import sqlite3  
from datetime import datetime
from flask import Flask, request, jsonify
import os


app = Flask(__name__)


folder_path = "model_trained_and_scalars/"
API_KEY = "9d3eea34a09240d74666495d538b519f"
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"
# ----------------------------------------------------------------------------
# configuration - paths to saved artifacts (adjust as needed)
MODEL_PATH = folder_path + "model18feb.pkl"
FEATURE_SCALER_PATH = folder_path + "feature_scaler18feb.pkl"
TARGET_SCALER_PATH = folder_path + "target_scaler18feb.pkl"

# ----------------------------------------------------------------------------
# load model and scalers at startup
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    model = None

try:
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)
except Exception:
    feature_scaler = None

try:
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)
except Exception:
    target_scaler = None


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

def fetch_weather(): #temp juggad
    """
    Fetch next 24-hour weather data from OpenWeather One Call 3.0 API.

    Output:
        list: 24 hourly weather dictionaries with:
              temp, dewpoint, humidity,
              windspeed, winddir, cloud, precip
    """

    # Example coordinates (Change if needed)
    lat = 28.6448   # Pune
    lon = 77.2167

    params = {
        "lat": lat,
        "lon": lon,
        "units": "metric",
        "exclude": "minutely,daily,alerts",
        "appid": API_KEY
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Weather API request failed: {e}")

    data = resp.json()

    hourly_data = data.get("hourly", [])[:24]

    if not hourly_data:
        raise ValueError("No hourly weather data found in API response")

    weather_features = []

    for hour in hourly_data:
        weather_features.append({
            "temp": hour.get("temp"),
            "dewpoint": hour.get("dew_point"),
            "humidity": hour.get("humidity"),
            "windspeed": hour.get("wind_speed"),
            "winddir": hour.get("wind_deg"),
            "cloud": hour.get("clouds"),
            "precip": hour.get("rain", {}).get("1h", 0.0)
        })

    return weather_features

def is_delhi_holiday(date_str):
    return date_str in delhi_holidays_2026


@app.route("/weather", methods=["GET"])
def weather():
    """
    Test route to check if weather API is working.
    Returns 24-hour weather forecast.
    """

    try:
        weather_data = fetch_weather()
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
    """Scale and reshape features for model input.
    
    Input:
        features (list): 2D list of shape (24, n_features)
                        24 hourly records, each with all feature values
    
    Output:
        np.array: Shape (1, 24, n_features) ready for model.predict()
    """
    arr = np.array(features, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 24:
        raise ValueError("Input must be 24 rows of features")

    if feature_scaler is not None:
        # flatten in the same way as during training
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



@app.route("/predict", methods=["POST"])
def predict():
    """Predict next 24-hour values from posted features.
    
    Input:
        POST JSON body: {"features": [[f1, f2, ...], ... , [f1, f2, ...]]}
                       - 24 lists of feature values
                       - Features in order: [hour, day, month, year, dayofweek, 
                                             temp, dewpoint, humidity, windspeed, 
                                             winddir, cloud, precip, is_holiday, is_weekend]
    
    Output:
        JSON: {"predictions": [val1, val2, ..., val24]}
              - 24 hourly prediction values (unscaled)
    """
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "missing 'features' in request"}), 400

    try:
        inp = preprocess(data["features"])
        pred_raw = model.predict(inp)
        result = postprocess(pred_raw)
        return jsonify({"predictions": result})
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    # optional debug flag; in production use gunicorn or similar
    app.run(host="0.0.0.0", port=5000, debug=True)


