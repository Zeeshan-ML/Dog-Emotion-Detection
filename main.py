# File: main.py
import os
import shutil
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import librosa
import numpy as np
import joblib
import uvicorn
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Dog Emotion Detection & Sound Generation API",
    description="Upload a dog audio file to predict its emotion and generate dog sounds.",
    version="2.0.0"
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

# Audio generation dependencies (lazy import to avoid startup issues)
audio_gen_available = False
try:
    from elevenlabs.client import ElevenLabs
    audio_gen_available = True
except ImportError:
    print("⚠️ ElevenLabs not installed. Audio generation endpoints will be disabled.")

# Pydantic model for audio generation request
class BarkRequest(BaseModel):
    """
    The request model for generating a sound effect.
    """
    prompt: str = Body(
        ...,
        examples=["a short, sharp bark", "a deep growl", "a happy woof"]
    )
    duration_seconds: Optional[float] = Body(
        None,
        examples=[3.0],
        description="Duration of the sound effect in seconds (0.5 to 22 seconds)."
    )
    prompt_influence: Optional[float] = Body(
        None,
        examples=[0.3],
        description="Optional prompt influence (0.0 to 1.0). Higher values are more creative."
    )

@app.on_event("startup")
async def startup_event():
    print("--- Loading ML models ---")
    if not os.path.exists(SCALER_PATH) or not os.path.exists(KMEANS_MODEL_PATH):
        raise RuntimeError(f"Model files not found in 'models/' folder.")

    try:
        models['scaler'] = joblib.load(SCALER_PATH)
        models['kmeans'] = joblib.load(KMEANS_MODEL_PATH)
        print("✅ Emotion detection models loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")
    
    # Initialize audio generation client if available
    if audio_gen_available:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            try:
                models['audio_client'] = ElevenLabs(api_key=api_key)
                print("✅ Audio generation client initialized.")
            except Exception as e:
                print(f"⚠️ Failed to initialize audio client: {e}")
        else:
            print("⚠️ ELEVENLABS_API_KEY not found in environment variables.")

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

# ============ ROOT ENDPOINT ============

@app.get("/")
async def root():
    return {
        "message": "Dog Emotion Detection & Sound Generation API",
        "version": "2.0.0",
        "endpoints": {
            "predict": "POST /predict/ - Upload dog audio to predict emotion",
            "generate_bark": "POST /generate-bark/ - Generate dog sound effects",
            "health": "GET /health - Check API health status"
        }
    }

# ============ AUDIO GENERATION ENDPOINTS ============

@app.post("/generate-bark/", summary="Generate Dog Sound Effect", description="Generate a dog sound effect based on a text prompt.")
async def generate_bark(request: BarkRequest):
    """
    This endpoint generates a sound effect from a text prompt.

    - **prompt**: A string describing the sound effect.
    - **prompt_influence** (optional): How strongly to follow the prompt (0.0 to 1.0).
    """
    if not audio_gen_available:
        raise HTTPException(
            status_code=503, 
            detail="Audio generation is not available. Install elevenlabs package."
        )
    
    if 'audio_client' not in models:
        raise HTTPException(
            status_code=503, 
            detail="Audio generation client is not initialized. Check your API key and server logs."
        )

    print(f"Received SFX prompt: {request.prompt}")

    try:
        client = models['audio_client']
        
        # Generate sound effect using ElevenLabs Sound Effects API
        # Note: Requires elevenlabs >= 1.9.0
        params = {"text": request.prompt}
        
        # Add optional parameters if provided
        if request.duration_seconds is not None:
            params["duration_seconds"] = request.duration_seconds
        if request.prompt_influence is not None:
            params["prompt_influence"] = request.prompt_influence
        
        audio_data = client.text_to_sound_effects.convert(**params)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Audio generation failed, no data returned.")
        
        print(f"Generated audio SFX for prompt: {request.prompt}")

        # Return as streaming response (audio_data is a generator)
        return StreamingResponse(
            audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=\"bark_sfx.mp3\""
            }
        )

    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@app.get("/health")
async def health_check():
    """Check the health status of all services."""
    return {
        "status": "healthy",
        "emotion_detection": "available" if models.get('scaler') and models.get('kmeans') else "unavailable",
        "audio_generation": "available" if audio_gen_available and 'audio_client' in models else "unavailable"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
