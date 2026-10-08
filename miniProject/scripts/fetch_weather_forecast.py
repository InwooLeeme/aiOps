"""
Open-Meteo "과거 예보(historical forecast)" 를 받아 data/weather_jeju_forecast.csv 로 저장한다.

관측값이 아니라 "그날 아침 시점의 예보값"이라서, 실서비스에서 받게 될 입력과 같은 종류다.
(관측 기상으로 평가하면 실서비스보다 성능이 낙관적으로 나온다.)

  python scripts/fetch_weather_forecast.py [start] [end]
"""
import sys
import time

import pandas as pd
import requests

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LAT, LON = 33.45, 126.55  # 제주 북부 저지대 (한라산 정상 격자 회피)
DAILY = ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean", "cloud_cover_mean",
         "shortwave_radiation_sum"]


def _get(start: str, end: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            r = requests.get(URL, timeout=120, params={
                "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end, "daily": ",".join(DAILY),
                "wind_speed_unit": "ms", "timezone": "Asia/Seoul"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  retry {k + 1}/{tries} {start}~{end}: {type(e).__name__}")
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"예보 API 실패: {start}~{end}")


def fetch(start: str, end: str) -> pd.DataFrame:
    """긴 구간은 연 단위로 나눠 요청한다(한 번에 받으면 타임아웃)."""
    parts = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        a, b = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        j = _get(a, b)
        parts.append(pd.DataFrame(j["daily"]).rename(columns={"time": "date"}))
        print(f"{y}: {len(parts[-1])}행 (elevation={j.get('elevation')}m)")
    df = pd.concat(parts, ignore_index=True)
    print(f"total rows={len(df)} nulls={df.isna().sum().to_dict()}")
    return df


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"
    out = fetch(start, end)
    out.to_csv("data/weather_jeju_forecast.csv", index=False)
    print(out.describe().round(2).T[["mean", "min", "max"]])
