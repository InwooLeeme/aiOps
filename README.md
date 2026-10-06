# AI Ops API

Python 3.14와 FastAPI로 구성한 API 프로젝트입니다. `uv`로 의존성과 가상환경을 관리합니다.

## 설치

프로젝트 폴더에서 실행합니다. `uv.lock`은 Git에 포함하여 동일한 의존성을 설치합니다.

```bash
uv sync --locked
```

`.venv` 가상환경이 자동으로 생성됩니다. `uv run`을 사용할 때는 별도로 가상환경을 활성화할 필요가 없습니다.

## 개발 서버 실행

```bash
uv run uvicorn app.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- 상태 확인: <http://127.0.0.1:8000/health>

`/health`는 서버가 응답하는지 확인하는 API이며 `{"status":"ok"}`를 반환합니다.

## 코드 검사

```bash
uv run ruff check .
uv run ruff format --check .
```

## 구조

```text
app/
  __init__.py
  main.py          # FastAPI 앱과 기본 API
pyproject.toml     # 프로젝트 및 의존성 설정
uv.lock            # 의존성 버전 고정
```

## 참고

- [FastAPI 시작하기](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [uv 프로젝트 관리](https://docs.astral.sh/uv/guides/projects/)
