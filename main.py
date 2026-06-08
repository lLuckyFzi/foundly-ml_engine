from fastapi import FastAPI
from api.routes import router as matching_router

app = FastAPI(title="Foundly ML Engine API", version="0.02")

app.include_router(matching_router, prefix="/api")

print("-> Server FastAPI siap menerima request!")