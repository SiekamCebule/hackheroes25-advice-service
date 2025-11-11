from fastapi import FastAPI

app = FastAPI(title="Advice API")


@app.get("/")
def root():
    return {"msg": "Hello HackHeroes!"}
