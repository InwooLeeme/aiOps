"""서버가 들고 있는 일별 테이블(최신 업로드 CSV + 예보 CSV). 파일이 바뀌면 다시 읽는다."""
import os
from threading import Lock

import pandas as pd

from data.features import WEATHER_CSV, load_table
from data.storage import latest_upload

_cache: dict = {}
_lock = Lock()


def get_table() -> pd.DataFrame:
    path = latest_upload()
    key = (path, os.path.getmtime(path), os.path.getmtime(WEATHER_CSV))
    with _lock:
        if _cache.get("key") != key:
            _cache.update(key=key, df=load_table(path))
        return _cache["df"]
