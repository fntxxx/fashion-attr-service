from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import CLIPModel, CLIPProcessor

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
        "mode": "imports-only",
        "torch_version": torch.__version__,
        "model_id": "openai/clip-vit-base-patch32",
        "endpoints": ["/health"],
    }

@app.get("/health")
def health():
    return {"ok": True}