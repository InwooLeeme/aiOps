"""테스트용 완전한 일별 관측값."""

from datetime import datetime, timedelta


def daily_rows(count=80, start="2022-01-01T00:00:00"):
    first = datetime.fromisoformat(start)
    return [
        {
            "timestamp": (first + timedelta(days=i)).isoformat(),
            "region": "제주",
            "generation_mwh": float(i % 24),
            "capacity_mw": 200.0,
            "temperature": -2.0,
            "humidity": 60.0,
            "wind_speed": 2.5,
            "cloud_cover": 3.0,
        }
        for i in range(count)
    ]
