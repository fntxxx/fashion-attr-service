from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "fashion-attr-service",
        "ok": True,
        "mode": "minimal",
        "endpoints": ["/health"],
    }

@app.get("/health")
def health():
    return {"ok": True}