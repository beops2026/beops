import pickle
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta
import urllib3
from flask import Flask, request, jsonify
import os
import json
import requests_cache
import openmeteo_requests
import pandas as pd
from retry_requests import retry
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
# Hugging Face hosted open model config (free tier / depends on provider availability)
# Set HF_TOKEN in your environment.
HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "Qwen/Qwen2.5-7B-Instruct"
)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

HF_TIMEOUT_SECONDS = 120

print("HF TOKEN LOADED :", bool(HF_TOKEN))
print("HF MODEL :", HF_MODEL_NAME)

# ----------------------------------------------------------------------------
# load model and scalers at startup
try:
    print("Loading model from:", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None

try:
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)
except Exception:
    target_scaler = None

try:
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)
    print("Feature scaler loaded")
except Exception as e:
    print("FEATURE SCALER LOAD ERROR:", e)
    feature_scaler = None


# ---------- helper utilities ----------
delhi_holidays_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-14",  # Makar Sankranti / Pongal
    "2026-01-26",  # Republic Day
    "2026-03-04",  # Maha Shivaratri
    "2026-03-21",  # Holi
    "2026-03-31",  # Ramzan Id (Eid-ul-Fitr)
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


# -------------------------------------------------------------------------
# SINGLE DAY PREDICTION
# -------------------------------------------------------------------------
def predict_single_day(date_dt):
    start_date = date_dt.strftime("%Y-%m-%d")

    weather_data = fetch_weather(start_date, start_date)
    features = []

    base_time = date_dt.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    for i, w in enumerate(weather_data):
        dt = base_time + timedelta(hours=i)

        row = [
            dt.hour,
            dt.day,
            dt.month,
            dt.year,
            dt.weekday(),
            w["temp"],
            w["dewpoint"],
            w["humidity"],
            w["windspeed"],
            w["winddir"],
            w["cloud"],
            w["precip"],
            int(is_delhi_holiday(start_date)),
            int(is_weekend(dt))
        ]
        features.append(row)

    inp = preprocess(features)
    pred_raw = model.predict(inp)
    result = postprocess(pred_raw)

    # Remove NaN / Inf values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    response_data = []
    for i, value in enumerate(result):
        dt = base_time + timedelta(hours=i)
        response_data.append({
            "time": int(dt.timestamp()),
            "value": float(value)
        })

    return response_data


# -------------------------------------------------------------------------
# WEATHER FETCH
# -------------------------------------------------------------------------
def fetch_weather(start_date, end_date):
    lat = 52.52
    lon = 13.41

    cache_session = requests_cache.CachedSession(
        '.cache',
        expire_after=3600
    )

    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.2
    )

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
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        responses = openmeteo.weather_api(
            url,
            params=params,
            verify=False
        )
    except Exception as e:
        raise Exception(f"Weather API request failed: {e}")

    response = responses[0]
    hourly = response.Hourly()

    temp = hourly.Variables(0).ValuesAsNumpy()
    humidity = hourly.Variables(1).ValuesAsNumpy()
    dewpoint = hourly.Variables(2).ValuesAsNumpy()
    precip = hourly.Variables(3).ValuesAsNumpy()
    windspeed = hourly.Variables(4).ValuesAsNumpy()
    winddir = hourly.Variables(5).ValuesAsNumpy()
    cloud = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = pd.DataFrame({
        "temp": temp,
        "humidity": humidity,
        "dewpoint": dewpoint,
        "precip": precip,
        "windspeed": windspeed,
        "winddir": winddir,
        "cloud": cloud
    })

    hourly_data = hourly_data.fillna(0)
    weather_features = hourly_data.to_dict(orient="records")

    if not weather_features:
        raise ValueError("No hourly weather data found")

    return weather_features


def is_delhi_holiday(date_str):
    return date_str in delhi_holidays_2026


@app.route("/hourly-weather", methods=["GET"])
def hourly_weather():
    """Test route to check if weather API is working."""
    try:
        weather_data = fetch_weather("2026-03-20", "2026-03-20")
        return jsonify(weather_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def fetch_holidays(db_path, start_date, end_date):
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
    cur.execute(
        "SELECT date FROM holidays WHERE date >= ? AND date <= ?",
        (start_date, end_date)
    )
    holidays = {r[0] for r in cur.fetchall()}
    conn.close()
    return holidays


def is_weekend(dt):
    """Check if date is weekend (Sat/Sun)."""
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
    """Inverse scale predictions back to original range."""
    raw = pred.reshape(-1, 1)
    if target_scaler is not None:
        inv = target_scaler.inverse_transform(raw)
    else:
        inv = raw
    return inv.flatten().tolist()


# -------------------------------------------------------------------------
# REPORT HELPERS
# -------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default


def _epoch_to_dt(epoch_value):
    return datetime.fromtimestamp(int(epoch_value))


def _format_time_label(ts, granularity):
    dt = datetime.fromtimestamp(int(ts))
    if granularity == "hourly":
        return dt.strftime("%Y-%m-%d %H:00")
    if granularity == "daily":
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m")


def build_prediction_series(report_type, start_dt, end_dt):
    """Collect prediction data according to the user's selected report type.

    Returns:
        tuple(series, granularity, metadata)
    """
    report_type = (report_type or "hourly").strip().lower()

    if report_type == "hourly":
        if start_dt.date() != end_dt.date():
            raise ValueError("For hourly report, start_date and end_date must be the same date")
        series = predict_single_day(start_dt)
        metadata = {
            "aggregation": "hourly forecast",
            "source": "predict_single_day"
        }
        return series, "hourly", metadata

    daily_series = []
    monthly_map = {}
    current_day = start_dt

    while current_day <= end_dt:
        daily_predictions = predict_single_day(current_day)
        daily_total = sum(_safe_float(item.get("value")) for item in daily_predictions)

        day_start = current_day.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        daily_series.append({
            "time": int(day_start.timestamp()),
            "value": round(daily_total, 2)
        })

        month_key = day_start.strftime("%Y-%m")
        if month_key not in monthly_map:
            month_start = day_start.replace(day=1)
            monthly_map[month_key] = {
                "time": int(month_start.timestamp()),
                "value": 0.0
            }
        monthly_map[month_key]["value"] += daily_total

        current_day += timedelta(days=1)

    if report_type in ("weekly", "monthly"):
        metadata = {
            "aggregation": "daily totals",
            "source": "predict_single_day"
        }
        return daily_series, "daily", metadata

    if report_type == "yearly":
        monthly_series = []
        for month_key in monthly_map:
            monthly_series.append({
                "time": monthly_map[month_key]["time"],
                "value": round(monthly_map[month_key]["value"], 2)
            })

        metadata = {
            "aggregation": "monthly totals",
            "source": "predict_single_day"
        }
        return monthly_series, "monthly", metadata

    raise ValueError("Invalid report_type. Use hourly, weekly, monthly, or yearly")


def summarize_series(series):
    values = [
        _safe_float(item.get("value"))
        for item in series
        if np.isfinite(_safe_float(item.get("value")))
    ]

    if not values:
        raise ValueError("No valid values available for summarization")

    timestamps = [item.get("time") for item in series]

    peak_idx = int(np.argmax(values))
    low_idx = int(np.argmin(values))

    average_value = float(np.mean(values))
    total_value = float(np.sum(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    std_value = float(np.std(values))

    trend = "stable"
    if len(values) >= 2:
        delta = values[-1] - values[0]
        if delta > 0:
            trend = "rising"
        elif delta < 0:
            trend = "falling"

    high_threshold = average_value + std_value
    critical_threshold = average_value + (1.5 * std_value)

    high_points = []
    for idx, value in enumerate(values):
        if value >= high_threshold:
            high_points.append({
                "time": timestamps[idx],
                "label": _format_time_label(
                    timestamps[idx],
                    "hourly" if len(series) == 24 else "daily" if len(series) > 1 and len(series) <= 31 else "monthly"
                ),
                "value": round(value, 2)
            })

    summary = {
        "points": len(values),
        "total": round(total_value, 2),
        "average": round(average_value, 2),
        "minimum": round(min_value, 2),
        "maximum": round(max_value, 2),
        "std_dev": round(std_value, 2),
        "trend": trend,
        "peak": {
            "time": timestamps[peak_idx],
            "value": round(values[peak_idx], 2)
        },
        "trough": {
            "time": timestamps[low_idx],
            "value": round(values[low_idx], 2)
        },
        "thresholds": {
            "high": round(high_threshold, 2),
            "critical": round(critical_threshold, 2)
        },
        "high_points": high_points[:10]
    }

    return summary


def build_rule_based_report(report_context):
    summary = report_context["summary"]
    series = report_context["series"]
    report_type = report_context["report_type"]

    peak_value = summary["peak"]["value"]
    peak_label = _format_time_label(
        summary["peak"]["time"],
        "hourly" if report_type == "hourly" else "daily" if report_type in ("weekly", "monthly") else "monthly"
    )
    average_value = summary["average"]
    trend = summary["trend"]
    std_dev = summary["std_dev"]

    if report_type == "hourly":
        scope_line = "Hourly forecast review for the selected day."
    elif report_type == "weekly":
        scope_line = "Daily totals reviewed across the selected week range."
    elif report_type == "monthly":
        scope_line = "Daily totals reviewed across the selected month range."
    else:
        scope_line = "Monthly totals reviewed across the selected yearly range."

    risk_lines = []
    if peak_value >= average_value + std_dev:
        risk_lines.append(
            "Load is materially above the mean during the peak period, so transformer and feeder loading should be watched closely."
        )
    if trend == "rising":
        risk_lines.append(
            "The trend is rising, which suggests demand is strengthening toward the end of the period."
        )
    if summary["maximum"] >= summary["thresholds"]["critical"]:
        risk_lines.append(
            "A critical spike is present and may justify contingency readiness."
        )
    if not risk_lines:
        risk_lines.append("No major abnormality is visible from the current forecast pattern.")

    action_lines = [
        "Monitor the peak window more frequently and keep substation alarms enabled.",
        "Check spare capacity, transformer loading, and feeder balancing before the peak period.",
        "Keep contingency switching instructions ready in case demand rises faster than expected."
    ]

    if trend == "rising":
        action_lines.append("Prepare for incremental load growth by reviewing operational margins.")
    elif trend == "falling":
        action_lines.append("Use the lower-demand window to schedule maintenance or inspection tasks.")

    next_steps = [
        "Validate the forecast against the latest operational observations.",
        "Share the peak window and risk note with the duty operator.",
        "Review whether any maintenance or switching activity should be moved away from the peak time."
    ]

    risk_block = "\n- ".join(risk_lines)
    action_block = "\n- ".join(action_lines)
    next_steps_block = "\n- ".join(next_steps)

    text = (
        f"Executive Summary\n"
        f"{scope_line}\n"
        f"Peak value: {peak_value} at {peak_label}\n"
        f"Average value: {average_value}\n"
        f"Trend: {trend}\n"
        f"\n"
        f"Key Observations\n"
        f"- Forecast points analysed: {len(series)}\n"
        f"- Standard deviation: {std_dev}\n"
        f"- Peak period indicates the highest operational attention point.\n"
        f"\n"
        f"Operational Risks\n"
        f"- {risk_block}\n"
        f"\n"
        f"Plan of Action\n"
        f"- {action_block}\n"
        f"\n"
        f"Next Steps\n"
        f"- {next_steps_block}"
    )

    return text


def build_llm_prompt(report_context):
    summary = report_context["summary"]
    report_type = report_context["report_type"]
    granularity = report_context["granularity"]
    series = report_context["series"]
    substation_name = report_context.get("substation_name", "grid substation operator")

    prompt_payload = {
        "report_type": report_type,
        "granularity": granularity,
        "substation_name": substation_name,
        "summary": summary,
        "series": series[:50]
    }

    instructions = (
        "You are an expert grid substation operations assistant. "
        "Generate a concise, practical report for a grid substation operator. "
        "Use only the data provided below. Do not invent numbers. "
        "Focus on operational interpretation, risk, plan of action, and next steps. "
        "Return strict JSON with these keys: "
        "executive_summary, key_observations, operational_risks, plan_of_action, next_steps. "
        "Each of the last four keys must be a JSON array of strings. "
        "The executive_summary key must be a single string."
    )

    return instructions + "\n\nDATA:\n" + json.dumps(prompt_payload, indent=2)


def extract_json_object(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    return None


def generate_hf_report(report_context):
    if not HF_TOKEN:
        print("HF token missing")
        return None, "missing_hf_token"

    prompt = build_llm_prompt(report_context)

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert power grid and electrical "
                    "substation analyst."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
        HF_ROUTER_URL,
        headers=headers,
        json=payload,
        timeout=HF_TIMEOUT_SECONDS,
        verify=False     
)

        print("HF STATUS :", response.status_code)
        print("HF RESPONSE :", response.text)

        response.raise_for_status()

        body = response.json()

        raw_text = (
            body["choices"][0]["message"]["content"]
            .strip()
        )

        parsed = extract_json_object(raw_text)

        if parsed is not None:
            return parsed, "huggingface_router"

        return raw_text, "huggingface_router"

    except Exception as e:

        print("HF ERROR :", str(e))

        return None, str(e)
    

@app.route("/report", methods=["POST"])
def generate_report():
    """Generate an operator-focused report using forecast data and a hosted Llama model."""

    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    try:
        payload = request.get_json(silent=True) or {}

        report_type = (
            payload.get("report_type")
            or request.args.get("report_type")
            or "hourly"
        ).strip().lower()
        start_epoch = payload.get("start_date") or request.args.get("start_date")
        end_epoch = payload.get("end_date") or request.args.get("end_date")
        substation_name = (
            payload.get("substation_name")
            or request.args.get("substation_name")
            or "Grid Substation"
        )

        if not start_epoch or not end_epoch:
            return jsonify({"error": "start_date and end_date required"}), 400

        start_dt = _epoch_to_dt(start_epoch)
        end_dt = _epoch_to_dt(end_epoch)

        if start_dt > end_dt:
            return jsonify({"error": "start_date must be less than or equal to end_date"}), 400

        if report_type not in {"hourly", "weekly", "monthly", "yearly"}:
            return jsonify({"error": "report_type must be hourly, weekly, monthly, or yearly"}), 400

        series, granularity, metadata = build_prediction_series(report_type, start_dt, end_dt)
        summary = summarize_series(series)

        report_context = {
            "report_type": report_type,
            "granularity": granularity,
            "metadata": metadata,
            "substation_name": substation_name,
            "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "series": series,
            "summary": summary
        }

        llm_report, report_source = generate_hf_report(report_context)

        if llm_report is None:
            llm_report = build_rule_based_report(report_context)
            report_source = "rule_based_fallback"

        return jsonify({
            "status": "success",
            "report_type": report_type,
            "substation_name": substation_name,
            "period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "metadata": metadata,
            "summary": summary,
            "report_source": report_source,
            "llm_model": HF_MODEL_NAME if report_source == "huggingface_router" else None,
            "report": llm_report
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    ok = model is not None
    return jsonify({"status": "ok" if ok else "error", "model_loaded": ok})


# -------------------------------------------------------------------------
# HOURLY PREDICTION API
# -------------------------------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict_auto():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    try:
        start_epoch = request.args.get("start_date")
        end_epoch = request.args.get("end_date")

        if not start_epoch or not end_epoch:
            return jsonify({"error": "start_date and end_date required"}), 400

        start_dt = datetime.fromtimestamp(int(start_epoch))
        end_dt = datetime.fromtimestamp(int(end_epoch))

        if start_dt.date() != end_dt.date():
            return jsonify({"error": "For /predict start_date and end_date must be same"}), 400

        predictions = predict_single_day(start_dt)
        return jsonify({"status": "success", "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# WEEKLY API
# -------------------------------------------------------------------------
@app.route("/predict-weekly", methods=["GET"])
def predict_weekly():
    try:
        start_epoch = request.args.get("start_date")
        end_epoch = request.args.get("end_date")

        start_dt = datetime.fromtimestamp(int(start_epoch))
        end_dt = datetime.fromtimestamp(int(end_epoch))

        current_day = start_dt
        response_data = []

        while current_day <= end_dt:
            daily_predictions = predict_single_day(current_day)
            daily_total = sum(
                float(item["value"])
                for item in daily_predictions
                if np.isfinite(item["value"])
            )

            response_data.append({
                "time": int(
                    current_day.replace(
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0
                    ).timestamp()
                ),
                "value": round(daily_total, 2)
            })

            current_day += timedelta(days=1)

        return jsonify({"status": "success", "predictions": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# MONTHLY API
# -------------------------------------------------------------------------
@app.route("/predict-monthly", methods=["GET"])
def predict_monthly():
    try:
        start_epoch = request.args.get("start_date")
        end_epoch = request.args.get("end_date")

        start_dt = datetime.fromtimestamp(int(start_epoch))
        end_dt = datetime.fromtimestamp(int(end_epoch))

        current_day = start_dt
        monthly_total = 0

        while current_day <= end_dt:
            daily_predictions = predict_single_day(current_day)
            daily_total = sum(
                float(item["value"])
                for item in daily_predictions
                if np.isfinite(item["value"])
            )
            monthly_total += daily_total
            current_day += timedelta(days=1)

        response_data = [{
            "time": int(
                start_dt.replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0
                ).timestamp()
            ),
            "value": round(monthly_total, 2)
        }]

        return jsonify({"status": "success", "predictions": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# Quarterly API due to OpenMateo restictions
# -------------------------------------------------------------------------
@app.route("/predict-quarterly", methods=["GET"])
def predict_quarterly():

    try:
        start_epoch = request.args.get("start_date")
        end_epoch = request.args.get("end_date")

        if not start_epoch or not end_epoch:
            return jsonify({
                "error": "start_date and end_date required"
            }), 400

        start_dt = datetime.fromtimestamp(
            int(start_epoch)
        )

        end_dt = datetime.fromtimestamp(
            int(end_epoch)
        )

        # Open-Meteo forecast limitation
        max_days = 108

        if (end_dt - start_dt).days > max_days:
            return jsonify({
                "error": (
                    f"Maximum supported range is "
                    f"{max_days} days"
                )
            }), 400

        current_day = start_dt

        monthly_data = {}

        while current_day <= end_dt:

            daily_predictions = predict_single_day(
                current_day
            )

            daily_total = sum(
                float(item["value"])
                for item in daily_predictions
                if np.isfinite(item["value"])
            )

            month_key = current_day.strftime("%Y-%m")

            if month_key not in monthly_data:

                month_start = current_day.replace(
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0
                )

                monthly_data[month_key] = {
                    "time": int(month_start.timestamp()),
                    "value": 0
                }

            monthly_data[month_key]["value"] += daily_total

            current_day += timedelta(days=1)

        response_data = []

        for month_key in sorted(monthly_data.keys()):

            response_data.append({
                "time":
                monthly_data[month_key]["time"],

                "value":
                round(
                    monthly_data[month_key]["value"],
                    2
                )
            })

        return jsonify({
            "status": "success",
            "range_type": "quarterly",
            "predictions": response_data
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
