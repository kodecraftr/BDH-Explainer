import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Import the handler from our new utils file
from Backend.utils import BDHModelHandler

# --- 1. Initialize App & Model ---
app = FastAPI(title="BDH Model API", description="API for Baby Dragon Hypernetwork")

# Global model handler
handler = None

@app.on_event("startup")
async def startup_event():
    global handler
    print("Initializing Model Handler...")
    handler = BDHModelHandler()

# --- 2. Data Models ---
class TextRequest(BaseModel):
    text: str = Field(..., example="Once upon a time")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(20, ge=1, le=100)

class NeuronRequest(BaseModel):
    text: str
    layer: int = Field(..., description="Layer number (1-based index)")

# --- 3. Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "model": "BDH-Transformer"}

@app.post("/predict")
def predict_next_token(request: TextRequest):
    """
    Predicts next tokens.
    Uses logic from 'recalculate_predictions.py' integrated into the model handler.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        predictions = handler.predict(
            request.text, 
            temperature=request.temperature, 
            top_k=request.top_k
        )
        return {"input_text": request.text, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
def get_embeddings(request: TextRequest):
    """
    Returns normalized embeddings.
    Uses logic from 'demo.py' (save_embeddings).
    """
    try:
        # We default to 32 dimensions as per your extract_embeddings.py
        data = handler.get_embeddings(request.text, target_dim=32)
        return {"input_text": request.text, **data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neurons")
def get_neuron_activations(request: NeuronRequest):
    """
    Returns neuron activations for the specified layer.
    Uses logic from 'extract_neuron_values.py'.
    Returns activations for the *last token* of the input.
    """
    if request.layer < 1 or request.layer > 6:
        raise HTTPException(status_code=400, detail="Layer must be between 1 and 6")

    try:
        data = handler.capture_neurons(request.text, request.layer)
        if not data:
            raise HTTPException(status_code=500, detail="Failed to capture neuron data")
        
        return {"input_text": request.text, **data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("Backend.server:app", host="0.0.0.0", port=8000, reload=True)