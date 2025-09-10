import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os
import json

# --- Пути ---
INPUT_CSV = "Cities_Dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Загрузка ---
df = pd.read_csv(INPUT_CSV)

# --- Проверка схемы ---
REQUIRED_COLS = [
    "name", "state",
    "latitude_high", "longitude_high",
    "latitude_low", "longitude_low",
    "timezone", "population", "area_km2", "is_capital"
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Пропущены обязательные столбцы: {missing}")

# --- Приведение типов ---
df["population"] = pd.to_numeric(df["population"], errors="coerce").astype("Int64")
df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")
df["latitude_high"] = pd.to_numeric(df["latitude_high"], errors="coerce")
df["longitude_high"] = pd.to_numeric(df["longitude_high"], errors="coerce")
df["latitude_low"] = pd.to_numeric(df["latitude_low"], errors="coerce")
df["longitude_low"] = pd.to_numeric(df["longitude_low"], errors="coerce")
df["is_capital"] = df["is_capital"].astype(bool)

# --- Геофункции ---
EARTH_RADIUS_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def percent_error(high, low):
    if abs(high) < 1e-12:
        return 0.0
    return abs(low - high) / abs(high) * 100.0

# --- Столица страны (USA) ---
dc_lat, dc_lon = 38.9072, -77.0369

# --- Столицы штатов ---
state_capitals = df[df["is_capital"]].groupby("state").agg({
    "latitude_high": "first", "longitude_high": "first"
}).rename(columns={"latitude_high": "cap_lat", "longitude_high": "cap_lon"})

# --- Высокая точность ---
df_high = df[["name","state","latitude_high","longitude_high","timezone","population","area_km2","is_capital"]].copy()
df_high.rename(columns={"latitude_high":"lat", "longitude_high":"lon"}, inplace=True)
df_high["density"] = df_high["population"] / df_high["area_km2"]
df_high["dist_from_DC_km"] = df_high.apply(lambda row: haversine(dc_lat, dc_lon, row["lat"], row["lon"]), axis=1)
df_high = df_high.merge(state_capitals, on="state", how="left")
df_high["dist_from_state_capital_km"] = df_high.apply(lambda row: haversine(row["cap_lat"], row["cap_lon"], row["lat"], row["lon"]), axis=1)

# --- Низкая точность ---
df_low = df[["name","state","latitude_low","longitude_low","timezone","population","area_km2","is_capital"]].copy()
df_low.rename(columns={"latitude_low":"lat", "longitude_low":"lon"}, inplace=True)
df_low["density"] = df_low["population"] / df_low["area_km2"]
df_low["dist_from_DC_km"] = df_low.apply(lambda row: haversine(dc_lat, dc_lon, row["lat"], row["lon"]), axis=1)
df_low = df_low.merge(state_capitals, on="state", how="left")
df_low["dist_from_state_capital_km"] = df_low.apply(lambda row: haversine(row["cap_lat"], row["cap_lon"], row["lat"], row["lon"]), axis=1)

# --- Сравнение точности ---
compare_df = pd.DataFrame({
    "name": df_high["name"],
    "state": df_high["state"],
    "DC_high_km": df_high["dist_from_DC_km"],
    "DC_low_km": df_low["dist_from_DC_km"],
    "DC_pct_err": [percent_error(h, l) for h, l in zip(df_high["dist_from_DC_km"], df_low["dist_from_DC_km"])],
    "State_high_km": df_high["dist_from_state_capital_km"],
    "State_low_km": df_low["dist_from_state_capital_km"],
    "State_pct_err": [percent_error(h, l) for h, l in zip(df_high["dist_from_state_capital_km"], df_low["dist_from_state_capital_km"])]
})

# --- Максимально удалённые ---
farthest_from_dc = df_high.loc[df_high["dist_from_DC_km"].idxmax(), ["name", "state", "dist_from_DC_km"]]
farthest_from_state = df_high.loc[
    df_high.groupby("state")["dist_from_state_capital_km"].idxmax(),
    ["state", "name", "dist_from_state_capital_km"]
]

# --- Часовые пояса ---
timezones_per_state = df_high.groupby("state")["timezone"].unique().apply(list)
one_timezone = {state: len(tzs) == 1 for state, tzs in timezones_per_state.items()}

# --- Корреляция долгот и часовых поясов ---
tz_to_offset = {
    "UTC-8": -8, "UTC-7": -7, "UTC-6": -6, "UTC-5": -5,
    "UTC+0": 0, "UTC+1": 1, "UTC+2": 2, "UTC+3": 3
}
df_high["utc_offset"] = df_high["timezone"].map(tz_to_offset)
corr = np.corrcoef(df_high["lon"], df_high["utc_offset"])[0, 1] if df_high["utc_offset"].notna().all() else None

# --- Память ---
mem_high = df_high[["lat","lon"]].astype("float64").memory_usage(deep=True).sum()
mem_low  = df_low[["lat","lon"]].astype("float32").memory_usage(deep=True).sum()
mem_saving = (1 - mem_low / mem_high) * 100

# --- Сохранение результатов ---
df_high.to_csv(f"{OUTPUT_DIR}/cities_enriched_high.csv", index=False)
df_low.to_csv(f"{OUTPUT_DIR}/cities_enriched_low.csv", index=False)
compare_df.to_csv(f"{OUTPUT_DIR}/distances_precision_compare.csv", index=False)
farthest_from_state.to_csv(f"{OUTPUT_DIR}/farthest_from_state_capitals.csv", index=False)
df_high[["name","state","timezone"]].to_csv(f"{OUTPUT_DIR}/timezones_per_city.csv", index=False)

with open(f"{OUTPUT_DIR}/memory_comparison.json", "w", encoding="utf-8") as f:
    json.dump({
        "bytes_high_float64": int(mem_high),
        "bytes_low_float32": int(mem_low),
        "saving_percent": round(mem_saving, 2)
    }, f, indent=2)

with open(f"{OUTPUT_DIR}/analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "farthest_from_DC": farthest_from_dc.to_dict(),
        "farthest_from_state_capitals": farthest_from_state.set_index("state").to_dict(orient="index"),
        "states_with_one_timezone": one_timezone,
        "timezone_correlation": round(corr, 3) if corr is not None else "N/A"
    }, f, indent=2)
