import os
import secrets
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pickle
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# Load model and encoders
model = joblib.load("model.pkl")
area_encoder = joblib.load("area_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")

AREAS = list(area_encoder.classes_)
SEASONS = list(season_encoder.classes_)

# Load real dataset
df = pd.read_csv("Electricity_2.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.sort_values(['area', 'date']).reset_index(drop=True)
LATEST_DATE = df['date'].max()
LATEST_YEAR = int(LATEST_DATE.year)
LATEST_MONTH = int(LATEST_DATE.month)

# Precompute per-area aggregated stats from real data
AREA_STATS = {}
for area_name, grp in df.groupby('area'):
    latest_year = grp['year'].max()
    latest = grp[grp['year'] == latest_year]
    prev_year = grp[grp['year'] == latest_year - 1]
    latest_month_row = grp[(grp['year'] == LATEST_YEAR) & (grp['month'] == LATEST_MONTH)]
    if len(latest_month_row) == 0:
        latest_month_row = grp[grp['date'] == grp['date'].max()]

    latest_services = int(round(float(latest_month_row['services'].iloc[-1])))
    total_units = round(latest['units'].sum() / 1e6, 2)  # convert to MU
    avg_load = round(latest['load'].mean(), 2)
    peak_demand = round(latest['load'].max(), 2)

    if len(prev_year) > 0:
        prev_units = prev_year['units'].sum()
        curr_units = latest['units'].sum()
        growth = round(((curr_units - prev_units) / max(prev_units, 1)) * 100, 1)
    else:
        growth = 0.0

    AREA_STATS[area_name] = {
        "area": area_name,
        "connections": latest_services,
        "consumption": total_units,
        "peak_demand": peak_demand,
        "growth": growth,
        "load_demand": avg_load,
    }

# ─── Circle → Division → Area hierarchy (Telangana Electricity) ───
# Circles and their Divisions
CIRCLE_DIVISIONS = {
    "Hyderabad North": ["Secunderabad Division", "Malkajgiri Division", "Begumpet Division"],
    "Hyderabad South": ["Charminar Division", "Saidabad Division", "Rajendranagar Division"],
    "Hyderabad Central": ["Abids Division", "Ameerpet Division", "Khairatabad Division"],
    "Rangareddy": ["Shamshabad Division", "Medchal Division", "LB Nagar Division"],
    "Warangal": ["Warangal Urban Division", "Warangal Rural Division"],
    "Karimnagar": ["Karimnagar Division", "Jagtial Division", "Peddapalli Division"],
    "Nizamabad": ["Nizamabad Division", "Kamareddy Division", "Adilabad Division"],
    "Nalgonda": ["Nalgonda Division", "Suryapet Division", "Khammam Division"],
    "Mahabubnagar": ["Mahabubnagar Division", "Nagarkurnool Division", "Gadwal Division"],
}

CIRCLES = list(CIRCLE_DIVISIONS.keys())

# Distribute model areas across divisions deterministically
def _build_hierarchy():
    """Assign each area from the encoder to a circle/division."""
    all_divs = []
    for circ in CIRCLES:
        for div in CIRCLE_DIVISIONS[circ]:
            all_divs.append((circ, div))

    hierarchy = {c: {d: [] for d in CIRCLE_DIVISIONS[c]} for c in CIRCLES}
    sorted_areas = sorted(AREAS)
    for i, area in enumerate(sorted_areas):
        circ, div = all_divs[i % len(all_divs)]
        hierarchy[circ][div].append(area)
    return hierarchy

HIERARCHY = _build_hierarchy()

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "telangana@2025"

# Telangana district coordinates for map
DISTRICT_COORDS = {
    "Hyderabad": [17.385, 78.4867],
    "Warangal": [17.9784, 79.5941],
    "Karimnagar": [18.4386, 79.1288],
    "Nizamabad": [18.6725, 78.0940],
    "Khammam": [17.2473, 80.1514],
    "Nalgonda": [17.0583, 79.2671],
    "Mahabubnagar": [16.7488, 77.9855],
    "Adilabad": [19.6641, 78.5320],
    "Medak": [18.0531, 78.2603],
    "Rangareddy": [17.2228, 78.2872],
    "Suryapet": [17.1399, 79.6266],
    "Siddipet": [18.1019, 78.8521],
    "Jagtial": [18.7947, 78.9137],
    "Peddapalli": [18.6151, 79.3782],
    "Mancherial": [18.8685, 79.4363],
    "Kamareddy": [18.3220, 78.3340],
    "Jangaon": [17.7232, 79.1520],
    "Nagarkurnool": [16.4818, 78.3129],
    "Wanaparthy": [16.3625, 78.0651],
    "Jogulamba Gadwal": [16.2305, 77.8068],
    "Sangareddy": [17.6287, 78.0869],
    "Medchal": [17.6298, 78.4810],
    "Yadadri Bhuvanagiri": [17.5918, 78.9521],
    "Vikarabad": [17.3384, 77.9048],
    "Rajanna Sircilla": [18.3872, 78.8100],
    "Kumuram Bheem": [19.3351, 79.4620],
    "Bhadadri Kothagudem": [17.5542, 80.6194],
    "Jayashankar Bhupalpally": [18.4354, 79.9870],
    "Mulugu": [18.1900, 80.5372],
    "Narayanpet": [16.7447, 77.4950],
    "Mahabubabad": [17.5982, 80.0008],
}

# Get area statistics from real data
def get_area_stats(area_name):
    """Return real stats for a given area from the dataset."""
    if area_name in AREA_STATS:
        return AREA_STATS[area_name].copy()
    # Fallback for unknown areas
    return {
        "area": area_name,
        "connections": 0,
        "consumption": 0,
        "peak_demand": 0,
        "growth": 0.0,
        "load_demand": 0,
    }


def get_dashboard_summary():
    """Aggregate stats across all areas."""
    total_consumption = 0
    latest_month_df = df[(df['year'] == LATEST_YEAR) & (df['month'] == LATEST_MONTH)]
    total_connections = int(round(latest_month_df['services'].sum()))
    total_peak = 0
    for area in AREAS:
        stats = get_area_stats(area)
        total_consumption += stats["consumption"]
        total_peak += stats["peak_demand"]
    avg_growth = round(np.mean([get_area_stats(a)["growth"] for a in AREAS]), 1)
    return {
        "total_consumption": round(total_consumption, 2),
        "total_connections": total_connections,
        "peak_demand": round(total_peak, 2),
        "avg_growth": avg_growth,
    }


# Telangana typical monthly climate profiles
MONTH_PROFILES = {
    1:  {"season": "Winter", "avg_temp": 23.5, "mintemp": 14.0, "maxtemp": 29.0, "rain": 5.0,  "minhumidity": 45, "maxhumidity": 72},
    2:  {"season": "Winter", "avg_temp": 26.5, "mintemp": 16.0, "maxtemp": 33.0, "rain": 8.0,  "minhumidity": 40, "maxhumidity": 68},
    3:  {"season": "Summer", "avg_temp": 31.5, "mintemp": 20.0, "maxtemp": 38.0, "rain": 10.0, "minhumidity": 35, "maxhumidity": 62},
    4:  {"season": "Summer", "avg_temp": 35.5, "mintemp": 24.0, "maxtemp": 41.0, "rain": 15.0, "minhumidity": 30, "maxhumidity": 58},
    5:  {"season": "Summer", "avg_temp": 36.0, "mintemp": 24.5, "maxtemp": 41.5, "rain": 25.0, "minhumidity": 35, "maxhumidity": 65},
    6:  {"season": "Rainy",  "avg_temp": 30.0, "mintemp": 22.0, "maxtemp": 35.0, "rain": 110.0,"minhumidity": 60, "maxhumidity": 88},
    7:  {"season": "Rainy",  "avg_temp": 27.5, "mintemp": 21.0, "maxtemp": 31.0, "rain": 155.0,"minhumidity": 70, "maxhumidity": 92},
    8:  {"season": "Rainy",  "avg_temp": 27.0, "mintemp": 21.0, "maxtemp": 30.0, "rain": 140.0,"minhumidity": 72, "maxhumidity": 92},
    9:  {"season": "Rainy",  "avg_temp": 27.0, "mintemp": 21.0, "maxtemp": 30.0, "rain": 105.0,"minhumidity": 68, "maxhumidity": 90},
    10: {"season": "Winter", "avg_temp": 26.5, "mintemp": 19.0, "maxtemp": 31.0, "rain": 65.0, "minhumidity": 55, "maxhumidity": 82},
    11: {"season": "Winter", "avg_temp": 23.5, "mintemp": 15.0, "maxtemp": 29.0, "rain": 25.0, "minhumidity": 48, "maxhumidity": 76},
    12: {"season": "Winter", "avg_temp": 21.0, "mintemp": 13.0, "maxtemp": 27.0, "rain": 8.0,  "minhumidity": 45, "maxhumidity": 73},
}


def run_forecast_2026(area, services_increase=0):
    """Forecast Jan-Jun 2026 using real historical lags from CSV data."""
    area_enc = int(area_encoder.transform([area])[0])
    area_df = df[df['area'] == area].sort_values('date')
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    base_services = float(AREA_STATS[area]['connections']) + services_increase
    base_load = float(AREA_STATS[area]['load_demand'])

    # Build per-service unit history for lags (model was trained on per-service units)
    per_service_units = list(area_df['units'].values / np.maximum(area_df['services'].values, 1))

    forecasts = []
    fc_labels = []
    lag_history = list(per_service_units)

    for i in range(6):
        month = i + 1  # Jan=1 ... Jun=6
        year = 2026
        p = MONTH_PROFILES[month]
        season_enc = int(season_encoder.transform([p['season']])[0])

        mintemp = p['mintemp']; maxtemp = p['maxtemp']; avg_temp = p['avg_temp']
        rain = p['rain']; temp_range = maxtemp - mintemp
        rain_log = float(np.log1p(rain))
        temp_rain_interaction = avg_temp * rain
        month_sin = float(np.sin(2 * np.pi * month / 12))
        month_cos = float(np.cos(2 * np.pi * month / 12))
        time_index = (year - 2019) * 12 + month

        n = len(lag_history)
        lag_1 = lag_history[-1]  if n >= 1  else 150.0
        lag_2 = lag_history[-2]  if n >= 2  else 150.0
        lag_3 = lag_history[-3]  if n >= 3  else 150.0
        lag_6 = lag_history[-6]  if n >= 6  else 150.0
        lag_12 = lag_history[-12] if n >= 12 else 150.0
        rolling_mean_3 = float(np.mean(lag_history[-3:] if n >= 3 else [150.0]))

        features = [
            area_enc, base_services, base_load, month, year, season_enc,
            mintemp, maxtemp, p['minhumidity'], p['maxhumidity'], rain,
            time_index, month_sin, month_cos,
            avg_temp, temp_range, rain_log, temp_rain_interaction,
            lag_1, lag_2, lag_3, lag_6, lag_12, rolling_mean_3
        ]

        prediction = float(model.predict([features])[0])
        lag_history.append(prediction)
        total_units = round(prediction * base_services, 2)
        forecasts.append(total_units)
        fc_labels.append(f"{month_names[month - 1]} 2026")

    return forecasts, fc_labels


def run_forecast(area, season, temperature, population):
    """Run 6-month forecast using loaded ML model."""
    area_enc = int(area_encoder.transform([area])[0])
    season_enc = int(season_encoder.transform([season])[0])

    base_month = {"Summer": 4, "Rainy": 7, "Winter": 11}
    start_month = base_month.get(season, 1)

    forecasts = []
    months_labels = []
    prev_predictions = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for i in range(6):
        month = ((start_month + i - 1) % 12) + 1
        year = 2025 + (start_month + i - 1) // 12

        month_sin = float(np.sin(2 * np.pi * month / 12))
        month_cos = float(np.cos(2 * np.pi * month / 12))

        temp_variation = 3 * np.sin(2 * np.pi * (month - 5) / 12)
        avg_temp = temperature + temp_variation
        mintemp = avg_temp - 4
        maxtemp = avg_temp + 4
        temp_range = maxtemp - mintemp
        minhumidity = 40.0
        maxhumidity = 75.0
        rain = 50.0 if season == "Rainy" or month in [6, 7, 8, 9] else 10.0
        rain_log = float(np.log1p(rain))
        temp_rain_interaction = avg_temp * rain

        services = population / 4.0
        load = services * 1.5
        # Use real data if available
        if area in AREA_STATS:
            services = float(AREA_STATS[area]['connections'])
            load = float(AREA_STATS[area]['load_demand'])

        time_index = (year - 2019) * 12 + month

        base_usage = 150.0
        lag_1 = prev_predictions[-1] if len(prev_predictions) >= 1 else base_usage
        lag_2 = prev_predictions[-2] if len(prev_predictions) >= 2 else base_usage
        lag_3 = prev_predictions[-3] if len(prev_predictions) >= 3 else base_usage
        lag_6 = base_usage
        lag_12 = base_usage

        recent = prev_predictions[-3:] if len(prev_predictions) >= 3 else [base_usage]
        rolling_mean_3 = float(np.mean(recent))

        features = [
            area_enc, services, load, month, year, season_enc,
            mintemp, maxtemp, minhumidity, maxhumidity, rain,
            time_index, month_sin, month_cos,
            avg_temp, temp_range, rain_log, temp_rain_interaction,
            lag_1, lag_2, lag_3, lag_6, lag_12, rolling_mean_3
        ]

        prediction = float(model.predict([features])[0])
        prev_predictions.append(prediction)

        total_units = prediction * services
        forecasts.append(round(total_units, 2))
        months_labels.append(f"{month_names[month - 1]} {year}")

    return forecasts, months_labels


# ─── Auth helpers ───────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.before_request
def require_login_for_protected_routes():
    """Ensure protected pages are not reachable without login."""
    endpoint = request.endpoint
    if endpoint is None:
        return None

    # Public endpoints.
    if endpoint in {"login", "index", "static"}:
        return None

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return None


# ─── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    summary = get_dashboard_summary()
    circle_area_map = {}
    hierarchy_map = {}
    for circle_name, divisions in HIERARCHY.items():
        area_list = []
        hierarchy_map[circle_name] = {}
        for _, div_areas in divisions.items():
            area_list.extend(div_areas)
        for div_name, div_areas in divisions.items():
            hierarchy_map[circle_name][div_name] = sorted(set(div_areas))
        circle_area_map[circle_name] = sorted(set(area_list))

    # Keep a 10-card dashboard view: 9 circles + one statewide aggregate.
    display_circles = list(CIRCLES) + ["All Telangana"]
    circle_area_map["All Telangana"] = sorted(AREAS)
    all_telangana_divisions = {}
    for circle_name, divisions in HIERARCHY.items():
        for div_name, div_areas in divisions.items():
            key = f"{circle_name} - {div_name}"
            all_telangana_divisions[key] = sorted(set(div_areas))
    hierarchy_map["All Telangana"] = all_telangana_divisions

    return render_template(
        "dashboard.html",
        summary=summary,
        areas=AREAS,
        circles=display_circles,
        circle_area_map=circle_area_map,
        hierarchy_map=hierarchy_map,
    )


@app.route("/analysis")
@login_required
def analysis():
    return render_template("analysis.html", areas=AREAS,
                           circles=CIRCLES, hierarchy=HIERARCHY)


@app.route("/api/area-stats")
@login_required
def api_area_stats():
    area = request.args.get("area", AREAS[0])
    if area not in AREAS:
        return jsonify({"error": "Unknown area"}), 400
    stats = get_area_stats(area)
    return jsonify(stats)


@app.route("/forecast")
@login_required
def forecast_page():
    return render_template("forecast.html", areas=AREAS, seasons=SEASONS,
                           circles=CIRCLES, hierarchy=HIERARCHY)


@app.route("/api/forecast-area", methods=["POST"])
@login_required
def api_forecast_area():
    data = request.get_json()
    area = data.get("area")
    services_increase = int(data.get("services_increase", 0))

    if area not in AREAS:
        return jsonify({"error": "Unknown area"}), 400

    # Historical monthly data for the area
    area_df = df[df['area'] == area].sort_values('date')
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hist_labels = []
    hist_units = []
    for _, row in area_df.iterrows():
        hist_labels.append(f"{month_names[int(row['month']) - 1]} {int(row['year'])}")
        hist_units.append(round(float(row['units']), 2))

    # Forecast Jan–Jun 2026
    fc_units, fc_labels = run_forecast_2026(area, services_increase)

    if len(fc_units) >= 2:
        change = ((fc_units[-1] - fc_units[0]) / max(fc_units[0], 1)) * 100
        trend = "Increasing" if change > 2 else ("Decreasing" if change < -2 else "Stable")
    else:
        trend, change = "Stable", 0

    return jsonify({
        "hist_labels": hist_labels,
        "hist_units": hist_units,
        "fc_labels": fc_labels,
        "fc_units": fc_units,
        "trend": trend,
        "change_pct": round(change, 1),
    })


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json()
    area = data.get("area")
    season = data.get("season")
    temperature = float(data.get("temperature", 30))
    population = float(data.get("population", 50000))

    if area not in AREAS:
        return jsonify({"error": "Unknown area"}), 400
    if season not in SEASONS:
        return jsonify({"error": "Unknown season"}), 400

    forecasts, labels = run_forecast(area, season, temperature, population)

    # Determine trend
    if len(forecasts) >= 2:
        change = ((forecasts[-1] - forecasts[0]) / max(forecasts[0], 1)) * 100
        trend = "Increasing" if change > 2 else ("Decreasing" if change < -2 else "Stable")
    else:
        trend = "Stable"
        change = 0

    return jsonify({
        "forecasts": forecasts,
        "labels": labels,
        "trend": trend,
        "change_pct": round(change, 1),
    })


@app.route("/maps")
@login_required
def maps():
    return render_template("maps.html", areas=AREAS, coords=DISTRICT_COORDS)


@app.route("/api/map-data")
@login_required
def api_map_data():
    """Return all district data for the map."""
    data = []
    for area in AREAS:
        stats = get_area_stats(area)
        coord = DISTRICT_COORDS.get(area, [17.385, 78.4867])
        stats["lat"] = coord[0]
        stats["lng"] = coord[1]
        data.append(stats)
    return jsonify(data)


@app.route("/reports")
@login_required
def reports():
    return render_template("reports.html", areas=AREAS, circles=CIRCLES,
                           hierarchy=HIERARCHY)


@app.route("/api/report", methods=["POST"])
@login_required
def api_report():
    data = request.get_json()
    area = data.get("area")

    if area not in AREAS:
        return jsonify({"error": "Unknown area"}), 400

    stats = get_area_stats(area)
    forecasts, labels = run_forecast_2026(area, 0)

    if len(forecasts) >= 2:
        change = ((forecasts[-1] - forecasts[0]) / max(forecasts[0], 1)) * 100
        trend = "Increasing" if change > 2 else ("Decreasing" if change < -2 else "Stable")
    else:
        trend = "Stable"
        change = 0

    report = {
        "area": area,
        "current_demand": stats["consumption"],
        "connections": stats["connections"],
        "peak_demand": stats["peak_demand"],
        "growth": stats["growth"],
        "forecasts": forecasts,
        "labels": labels,
        "trend": trend,
        "change_pct": round(change, 1),
    }
    return jsonify(report)


@app.route("/api/monthly-trend")
@login_required
def api_monthly_trend():
    """Return monthly Telangana consumption from Jan 2021 to Dec 2025."""
    trend_df = df[(df['year'] >= 2021) & (df['year'] <= 2025)].copy()
    trend_df = trend_df.sort_values(['year', 'month'])

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grouped = (
        trend_df.groupby(['year', 'month'], as_index=False)['units']
        .sum()
        .sort_values(['year', 'month'])
    )

    months = [f"{month_names[int(r.month) - 1]} {int(r.year)}" for r in grouped.itertuples()]
    values = [round(float(r.units) / 1e6, 2) for r in grouped.itertuples()]
    return jsonify({"months": months, "values": values})


# ─── Hierarchy API endpoints ───────────────────────────────
@app.route("/api/circles")
@login_required
def api_circles():
    return jsonify(CIRCLES)


@app.route("/api/divisions/<circle>")
@login_required
def api_divisions(circle):
    if circle not in HIERARCHY:
        return jsonify({"error": "Unknown circle"}), 400
    return jsonify(list(HIERARCHY[circle].keys()))


@app.route("/api/areas/<circle>/<division>")
@login_required
def api_areas(circle, division):
    if circle not in HIERARCHY:
        return jsonify({"error": "Unknown circle"}), 400
    if division not in HIERARCHY[circle]:
        return jsonify({"error": "Unknown division"}), 400
    return jsonify(HIERARCHY[circle][division])


@app.route("/api/area-details/<area_name>")
@login_required
def api_area_details(area_name):
    """Return stats + hierarchy info for a specific area."""
    if area_name not in AREAS:
        return jsonify({"error": "Unknown area"}), 400
    stats = get_area_stats(area_name)
    # Find circle and division for this area
    for circ, divs in HIERARCHY.items():
        for div, areas_list in divs.items():
            if area_name in areas_list:
                stats["circle"] = circ
                stats["division"] = div
                break
    coords = DISTRICT_COORDS.get(area_name)
    if coords:
        stats["lat"] = coords[0]
        stats["lng"] = coords[1]
    return jsonify(stats)


@app.route("/api/area-trends/<area_name>")
@login_required
def api_area_trends(area_name):
    """Return historical trend data for an area from real dataset."""
    if area_name not in AREAS:
        return jsonify({"error": "Unknown area"}), 400

    area_df = df[df['area'] == area_name].copy()

    years = sorted(area_df['year'].unique().tolist())
    year_labels = [str(y) for y in years]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # --- Temperature trend (yearly avg) ---
    avg_temp_col = (area_df['mintemp'] + area_df['maxtemp']) / 2
    area_df['avg_temp'] = avg_temp_col
    temp_yearly = []
    for y in years:
        yr_data = area_df[area_df['year'] == y]
        temp_yearly.append(round(yr_data['avg_temp'].mean(), 1))

    # --- Monthly temperature for latest year ---
    latest_year = max(years)
    latest_df = area_df[area_df['year'] == latest_year]
    monthly_temps = []
    for m in range(1, 13):
        m_data = latest_df[latest_df['month'] == m]
        if len(m_data) > 0:
            monthly_temps.append(round(m_data['avg_temp'].mean(), 1))
        else:
            monthly_temps.append(0)

    # --- Consumption trend (yearly, MU) ---
    consumption_yearly = []
    for y in years:
        yr_data = area_df[area_df['year'] == y]
        total_mu = round(yr_data['units'].sum() / 1e6, 2)
        consumption_yearly.append(total_mu)

    # --- Monthly consumption for latest year ---
    monthly_consumption = []
    for m in range(1, 13):
        m_data = latest_df[latest_df['month'] == m]
        if len(m_data) > 0:
            monthly_consumption.append(round(m_data['units'].sum() / 1e6, 2))
        else:
            monthly_consumption.append(0)

    # --- Services / Connections growth (yearly) ---
    services_yearly = []
    for y in years:
        yr_data = area_df[area_df['year'] == y]
        services_yearly.append(int(round(yr_data['services'].mean())))

    # --- Peak demand vs load demand (yearly, MW) ---
    peak_demands = []
    load_demands = []
    for y in years:
        yr_data = area_df[area_df['year'] == y]
        peak_demands.append(round(yr_data['load'].max(), 2))
        load_demands.append(round(yr_data['load'].mean(), 2))

    # --- Seasonal consumption breakdown (Summer, Rainy, Winter) ---
    latest_seasons = latest_df.groupby('season')['units'].sum()
    seasonal_consumption = {}
    for s in ['Summer', 'Rainy', 'Winter']:
        if s in latest_seasons.index:
            seasonal_consumption[s] = round(latest_seasons[s] / 1e6, 2)
        else:
            seasonal_consumption[s] = 0

    # --- Monthly rainfall for latest year ---
    rain_monthly = []
    for m in range(1, 13):
        m_data = latest_df[latest_df['month'] == m]
        if len(m_data) > 0:
            rain_monthly.append(round(m_data['rain'].mean(), 1))
        else:
            rain_monthly.append(0)

    return jsonify({
        "years": year_labels,
        "months": month_names,
        "temp_yearly": temp_yearly,
        "temp_monthly": monthly_temps,
        "consumption_yearly": consumption_yearly,
        "consumption_monthly": monthly_consumption,
        "services_yearly": services_yearly,
        "peak_demand_yearly": peak_demands,
        "load_demand_yearly": load_demands,
        "seasonal_consumption": seasonal_consumption,
        "rain_monthly": rain_monthly,
    })


@app.route("/api/area-monthly-data/<area_name>")
@login_required
def api_area_monthly_data(area_name):
    """Return all monthly rows from the CSV for a given area."""
    if area_name not in AREAS:
        return jsonify({"error": "Unknown area"}), 400

    area_df = df[df['area'] == area_name].sort_values('date')

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Build monthly timeline labels and consumption values
    timeline_labels = []
    timeline_units = []
    rows = []

    for _, row in area_df.iterrows():
        label = f"{month_names[int(row['month']) - 1]} {int(row['year'])}"
        timeline_labels.append(label)
        timeline_units.append(round(float(row['units']), 2))
        rows.append({
            "month": label,
            "services": int(row['services']),
            "units": round(float(row['units']), 2),
            "load": round(float(row['load']), 2),
            "season": row['season'],
            "mintemp": round(float(row['mintemp']), 1),
            "maxtemp": round(float(row['maxtemp']), 1),
            "rain": round(float(row['rain']), 1),
        })

    # Last 6 months
    last6_labels = timeline_labels[-6:]
    last6_units = timeline_units[-6:]

    # Seasonal aggregation (all-time)
    seasonal = area_df.groupby('season')['units'].sum()
    seasonal_data = {}
    for s in ['Summer', 'Rainy', 'Winter']:
        seasonal_data[s] = round(float(seasonal.get(s, 0)), 2)

    return jsonify({
        "timeline_labels": timeline_labels,
        "timeline_units": timeline_units,
        "last6_labels": last6_labels,
        "last6_units": last6_units,
        "seasonal": seasonal_data,
        "rows": rows,
    })


if __name__ == "__main__":
    app.run(debug=True)
