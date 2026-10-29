"""Deterministic, explainable risk scoring.

The LLM never does the math: it reads these scores and their reasons.
Thresholds are a v1 heuristic, to be tuned with real users and evals.
Design rule: weather alone caps at 50 (advisory). Evacuations require a fire.
"""

LEVELS = [(85, "evacuation_order"), (60, "evacuation_warning"), (35, "advisory"), (0, "monitor")]


def score_zone(weather: dict, fires: list[dict]) -> dict:
    score, reasons = 0, []

    alerts = weather.get("alerts", [])
    if "Red Flag Warning" in alerts:
        score += 25; reasons.append("Red Flag Warning active (+25)")
    elif "Fire Weather Watch" in alerts:
        score += 10; reasons.append("Fire Weather Watch active (+10)")

    wind = weather.get("max_wind_mph") or 0
    if wind >= 25:
        score += 15; reasons.append(f"Peak wind {wind} mph >= 25 (+15)")
    elif wind >= 15:
        score += 8; reasons.append(f"Peak wind {wind} mph >= 15 (+8)")

    rh = weather.get("min_rh_pct")
    if rh is not None and rh <= 15:
        score += 10; reasons.append(f"Min humidity {rh}% <= 15 (+10)")
    elif rh is not None and rh <= 25:
        score += 5; reasons.append(f"Min humidity {rh}% <= 25 (+5)")

    if fires:
        d = fires[0]["distance_km"]
        if d <= 5:
            score += 55; reasons.append(f"Active fire {d} km away (+55)")
        elif d <= 15:
            score += 35; reasons.append(f"Active fire {d} km away (+35)")
        else:
            score += 15; reasons.append(f"Active fire {d} km away (+15)")

    score = min(score, 100)
    level = next(name for threshold, name in LEVELS if score >= threshold)
    return {"score": score, "recommended_level": level, "reasons": reasons or ["No significant risk factors"]}
