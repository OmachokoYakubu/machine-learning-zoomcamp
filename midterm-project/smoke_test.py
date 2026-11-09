#!/usr/bin/env python3
"""End-to-end HTTP smoke test for the BikePulse FastAPI service."""

from __future__ import annotations

import argparse
import json
import urllib.request


def http_json(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200, f"Unexpected status {resp.status} for {url}"
        return json.loads(resp.read().decode("utf-8"))


def main(base_url: str) -> None:
    base = base_url.rstrip("/")
    health = http_json(f"{base}/health")
    assert health["status"] == "ok", f"Unhealthy response: {health}"

    rush_hour = http_json(
        f"{base}/predict",
        {
            "season": 3,
            "yr": 1,
            "mnth": 9,
            "hr": 17,
            "holiday": 0,
            "weekday": 3,
            "workingday": 1,
            "weathersit": 1,
            "temp_celsius": 24.0,
            "humidity_pct": 50.0,
            "windspeed_kmh": 11.0,
        },
    )
    rainy_night = http_json(
        f"{base}/predict",
        {
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": 3,
            "holiday": 0,
            "weekday": 2,
            "workingday": 1,
            "weathersit": 3,
            "temp_celsius": 4.0,
            "humidity_pct": 88.0,
            "windspeed_kmh": 28.0,
        },
    )
    assert (
        rush_hour["predicted_hourly_rentals"] > rainy_night["predicted_hourly_rentals"]
    ), "Expected rush hour clear weather demand to exceed rainy 03:00 AM demand"

    batch = http_json(
        f"{base}/predict/batch",
        {
            "hours": [
                {"hr": h, "mnth": 9, "season": 3, "weekday": 3, "weathersit": 1, "temp_celsius": 22.0}
                for h in (3, 8, 12, 17)
            ]
        },
    )
    assert batch["horizon_hours"] == 4

    print(
        json.dumps(
            {
                "health": health,
                "rush_hour_forecast": rush_hour,
                "rainy_night_forecast": rainy_night,
                "batch_summary": {
                    "horizon_hours": batch["horizon_hours"],
                    "peak_hour": batch["peak_hour"],
                    "peak_predicted_rentals": batch["peak_predicted_rentals"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test a running BikePulse API")
    parser.add_argument("--url", default="http://127.0.0.1:9696")
    args = parser.parse_args()
    main(args.url)

