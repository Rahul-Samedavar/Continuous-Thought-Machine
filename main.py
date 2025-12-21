import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from app.model_loader import load_model
from app.inference import run_inference

# --------------------
# App setup
# --------------------
app = FastAPI(title="CTM Inference Server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve generated GIFs
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Load model ONCE
model = load_model("logs/custom_run/checkpoint_10000.pt")


# --------------------
# Routes
# --------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/inference")
async def inference_endpoint(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    temp_name = f"{uuid4()}.{ext}"
    temp_path = os.path.join(temp_name)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_inference(temp_path, model, OUTPUT_DIR)
    gif_filename = result.pop("gif_filename")
    result["gif_url"] = f"/outputs/{gif_filename}"

    return result


# --------------------
# Run directly
# --------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
