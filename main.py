# File: main.py
import os
import shutil
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import librosa
import numpy as np
import joblib
import uvicorn

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Dog Emotion Prediction API",
    description="Upload a dog audio file to predict its emotion.",
    version="1.0.0"
)

# Paths
SCALER_PATH = 'models/scaler.joblib'
KMEANS_MODEL_PATH = 'models/kmeans_model.joblib'

models = {}

CLUSTER_EMOTION_MAP = {
    0: "Angry/Aggressive",
    1: "Happy/Playful",
    2: "Anxious/Whining",
    3: "Neutral",
    4: "Fearful"
}

# Mount static files folder for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder for HTML
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    print("--- Loading ML models ---")
    if not os.path.exists(SCALER_PATH) or not os.path.exists(KMEANS_MODEL_PATH):
        raise RuntimeError(f"Model files not found in 'models/' folder.")

    try:
        models['scaler'] = joblib.load(SCALER_PATH)
        models['kmeans'] = joblib.load(KMEANS_MODEL_PATH)
        print("✅ Models loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")

def extract_features(audio_path, n_mfcc=40):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    if not models:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please try again later.")

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\nProcessing audio file: {file.filename}")
        features = extract_features(temp_file_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Feature extraction failed. Check the audio file format.")

        scaled_features = models['scaler'].transform(features.reshape(1, -1))
        predicted_cluster = models['kmeans'].predict(scaled_features)[0]
        predicted_emotion = CLUSTER_EMOTION_MAP.get(predicted_cluster, "Unknown Emotion")

        print(f"  -> Predicted Cluster: {predicted_cluster}")
        print(f"  -> Mapped Emotion: {predicted_emotion}")

        response_data = {
            "status": "success",
            "filename": file.filename,
            "cluster": int(predicted_cluster),
            "emotion": predicted_emotion
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
