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
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"


def fetch_weather():
    """
    Fetch next 24-hour weather data using OpenWeather Free Plan (2.5 forecast).

    Output:
        list: 8 records (3-hour interval covering 24 hours)
              Each containing:
              temp, dewpoint, humidity,
              windspeed, winddir, cloud, precip
    """

    lat = 28.6448   # Delhi (change if needed)
    lon = 77.2167

    params = {
        "lat": lat,
        "lon": lon,
        "units": "metric",
        "appid": API_KEY
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Weather API request failed: {e}")

    data = resp.json()

    forecast_list = data.get("list", []) [:8] # 8 records = 24 hours (3-hour interval)
    #remove [:8] toget 5 day forecast from current weather 

    if not forecast_list:
        raise ValueError("No forecast data found in API response")

    weather_features = []

    for item in forecast_list:
        weather_features.append({
            "temp": item.get("main", {}).get("temp"),
            "dewpoint": None,  # Not available in free API
            "humidity": item.get("main", {}).get("humidity"),
            "windspeed": item.get("wind", {}).get("speed"),
            "winddir": item.get("wind", {}).get("deg"),
            "cloud": item.get("clouds", {}).get("all"),
            "precip": item.get("rain", {}).get("3h", 0.0)
        })

    return weather_features

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
    

if __name__ == "__main__":
    # optional debug flag; in production use gunicorn or similar
    app.run(host="0.0.0.0", port=5000, debug=True)


