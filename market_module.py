"""
market_module.py — Weather + mandi price advisory logic for crop selling decisions.

The response always includes a live data card (weather + price) followed by
the LLM's sell/wait/store recommendation, making it clear what data drove the advice.
"""

import json
import os

import requests
from dotenv import load_dotenv

from llm import call_llm
from prompts import MARKET_PROMPT_TEMPLATE, SYSTEM_PROMPT_BASE

load_dotenv()

# Mapping from district names to OpenWeatherMap-compatible city strings
DISTRICT_TO_CITY = {
    "pune": "Pune,IN",
    "nashik": "Nashik,IN",
    "aurangabad": "Aurangabad,IN",
    "nagpur": "Nagpur,IN",
    "kolhapur": "Kolhapur,IN",
    "mumbai": "Mumbai,IN",
    "solapur": "Solapur,IN",
    "satara": "Satara,IN",
    "sangli": "Sangli,IN",
    "latur": "Latur,IN",
    "nanded": "Nanded,IN",
    "amravati": "Amravati,IN",
    "akola": "Akola,IN",
    "jalgaon": "Jalgaon,IN",
    "ratnagiri": "Ratnagiri,IN",
}

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mandi_prices.json")

TREND_EMOJI = {"rising": "📈", "falling": "📉", "stable": "➡️"}


def get_weather(district_name: str) -> dict:
    """
    Fetch a 24-hour weather forecast for the given district from OpenWeatherMap.

    Returns a dict with:
        temp_c        — current temperature in Celsius
        humidity_pct  — current humidity percentage
        rain_expected — True if rain is forecast in the next ~24 hours
        description   — short human-readable sky condition
        wind_kmh      — wind speed in km/h
        source        — "live" or "fallback"
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    district_lower = district_name.strip().lower()
    city = DISTRICT_TO_CITY.get(district_lower, f"{district_name.capitalize()},IN")

    fallback = {
        "temp_c": 28,
        "humidity_pct": 60,
        "rain_expected": False,
        "description": "Clear sky",
        "wind_kmh": 10,
        "source": "fallback",
    }

    if not api_key or api_key == "your_openweathermap_key_here":
        print(
            "[market_module.py] OPENWEATHER_API_KEY not configured — using fallback weather."
        )
        return fallback

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={api_key}&units=metric&cnt=8"  # cnt=8 → next 24 hours (3h slots)
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        first = data["list"][0]
        temp_c = first["main"]["temp"]
        humidity_pct = first["main"]["humidity"]
        desc = first["weather"][0]["description"].capitalize()
        wind_ms = first.get("wind", {}).get("speed", 0)
        wind_kmh = round(wind_ms * 3.6, 1)

        # Rain expected if any of the next 8 slots (24h) have rain
        rain_expected = any(
            "rain" in slot["weather"][0]["main"].lower()
            or slot.get("rain", {}).get("3h", 0) > 0
            for slot in data["list"]
        )

        print(
            f"[market_module.py] Live weather for {city}: "
            f"{temp_c}°C, {desc}, rain={rain_expected}"
        )

        return {
            "temp_c": round(temp_c, 1),
            "humidity_pct": humidity_pct,
            "rain_expected": rain_expected,
            "description": desc,
            "wind_kmh": wind_kmh,
            "source": "live",
        }

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        print(f"[market_module.py] Weather API HTTP {status} for '{city}': {e}")
        if status == 404:
            print(
                f"  → City '{city}' not found on OpenWeatherMap. Add it to DISTRICT_TO_CITY."
            )
        return fallback
    except Exception as e:
        print(f"[market_module.py] Weather fetch error: {e}")
        return fallback


def get_mandi_price(crop: str, district: str) -> dict | None:
    """
    Look up the mandi price for a crop + district from local JSON data.
    Returns a dict {price_per_quintal, trend, unit} or None if not found.
    """
    crop = crop.strip().lower()
    district = district.strip().lower()

    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[market_module.py] Failed to load mandi_prices.json: {e}")
        return None

    district_data = data.get(district)
    if not district_data:
        print(f"[market_module.py] District '{district}' not in mandi data.")
        return None

    crop_data = district_data.get(crop)
    if not crop_data:
        print(f"[market_module.py] Crop '{crop}' not in '{district}' mandi data.")
        return None

    return crop_data


def _extract_crop_and_district(query: str) -> tuple[str, str]:
    """Use the LLM to extract crop and district from a natural language query."""
    extraction_prompt = (
        "Extract the crop name and district/city name from the following farmer query. "
        'Return ONLY a valid JSON object with two keys: "crop" and "district". '
        "Normalise the crop name to English (e.g. 'pyaaz' → 'onion', 'tamatar' → 'tomato'). "
        "If a value cannot be determined, use an empty string. No explanation.\n\n"
        f"Query: {query}"
    )
    raw = ""
    try:
        raw = call_llm("You are a JSON extraction assistant.", extraction_prompt)
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        crop = str(parsed.get("crop", "")).strip().lower()
        district = str(parsed.get("district", "")).strip().lower()
        return crop, district
    except Exception as e:
        print(f"[market_module.py] Failed to parse crop/district: {e} | raw='{raw}'")
        return "", ""


def _build_data_card(
    crop: str,
    district: str,
    weather: dict,
    price_data: dict | None,
) -> str:
    """
    Build a formatted data card showing live weather + price data.
    This is prepended to every market response so the farmer can see
    exactly what data drove the recommendation.
    """
    # Weather section
    rain_line = (
        "⛈️  Rain expected in next 24 hours!"
        if weather["rain_expected"]
        else "☀️  No rain expected in next 24 hours"
    )
    weather_source = (
        "🔴 Live" if weather["source"] == "live" else "⚪ Estimated (no API key)"
    )

    weather_block = (
        f"🌦️  {weather['description']}  |  🌡️ {weather['temp_c']}°C  "
        f"|  💧 {weather['humidity_pct']}% humidity  |  💨 {weather['wind_kmh']} km/h\n"
        f"{rain_line}\n"
        f"Weather data: {weather_source}"
    )

    # Price section
    if price_data:
        trend_emoji = TREND_EMOJI.get(price_data["trend"], "➡️")
        price_block = (
            f"💰  ₹{price_data['price_per_quintal']:,} per quintal ({price_data['unit']})  "
            f"|  {trend_emoji} Trend: {price_data['trend'].upper()}"
        )
    else:
        price_block = "💰  Price data not available for this crop/district combination."

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊  {crop.capitalize()} Market — {district.capitalize()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{weather_block}\n"
        f"{price_block}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def handle_market_query(text_query: str) -> str:
    """
    Handle a market/selling advisory query.

    Steps:
    1. Extract crop + district from the query.
    2. Fetch live weather from OpenWeatherMap.
    3. Look up mandi price from local JSON.
    4. Build a visible data card (always shown).
    5. Ask LLM for a sell/wait/store recommendation.
    6. Return data card + LLM recommendation combined.
    """
    crop, district = _extract_crop_and_district(text_query)

    if not crop or not district:
        return (
            "I could not identify the crop or district from your question.\n"
            "Please mention both — for example: 'Should I sell my onions in Mumbai today?'"
        )

    # Fetch live data
    weather = get_weather(district)
    price_data = get_mandi_price(crop, district)

    # Build the data card (always visible in the response)
    data_card = _build_data_card(crop, district, weather, price_data)

    # Build strings for the LLM prompt
    rain_note = (
        "Yes — rain is expected in the next 24 hours. Transport and storage risks are HIGH."
        if weather["rain_expected"]
        else "No rain expected. Conditions are good for transport."
    )
    weather_str = (
        f"Temperature: {weather['temp_c']}°C | "
        f"Humidity: {weather['humidity_pct']}% | "
        f"Wind: {weather['wind_kmh']} km/h | "
        f"Conditions: {weather['description']} | "
        f"Rain in next 24h: {rain_note}"
    )

    if price_data:
        price_str = (
            f"Current price: Rs.{price_data['price_per_quintal']} per quintal | "
            f"Trend: {price_data['trend'].upper()}"
        )
    else:
        price_str = (
            f"No mandi price data available for {crop} in {district}. "
            "Advise the farmer to check locally."
        )

    filled_prompt = MARKET_PROMPT_TEMPLATE.format(
        crop=crop.capitalize(),
        district=district.capitalize(),
        price_data=price_str,
        weather_data=weather_str,
    )

    recommendation = call_llm(SYSTEM_PROMPT_BASE, filled_prompt)

    # Combine data card + LLM recommendation
    return f"{data_card}\n\n{recommendation}"
