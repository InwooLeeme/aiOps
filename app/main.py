from fastapi import FastAPI

app = FastAPI(
    title="AI Ops API",
    description="AI Ops 서비스 API",
    version="0.1.0",
)


@app.get("/", tags=["default"])
async def root() -> dict[str, str]:
    return {"message": "AI Ops API", "docs": "/docs"}


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
