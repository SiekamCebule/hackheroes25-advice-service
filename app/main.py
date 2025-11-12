import logging

from fastapi import FastAPI

from app.routers.advice import router as advice_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Advice API")

app.include_router(advice_router)


@app.get("/")
def root():
    return {"msg": "Hello HackHeroes!"}
