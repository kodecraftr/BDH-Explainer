# BDH Visualizer - Backend

The Backend module acts as the bridging API between the PyTorch language models (BDH and Transformer) and the Next.js interactive 3D frontend. It is constructed using **FastAPI** to provide high-performance, asynchronous RESTful endpoints for model inferences and internal state extractions.

---

## 🚀 Quick Start

Ensure you have your Conda environment active (`conda activate ml`).

To launch the server locally:
```bash
python -m uvicorn Backend.server:app --host 0.0.0.0 --port 8000 --reload
```
*Alternatively, you can run the `start_server.sh` script located in the repository root.*

The API will be available at `http://localhost:8000`. You can view the automatically generated interactive API documentation (Swagger) by navigating to `http://localhost:8000/docs` in your browser.

---

## 📂 Architecture

### 1. `server.py`
The FastAPI application wrapper. It manages route definitions, Pydantic data validation schemas, and HTTP error handling. 

### 2. `utils.py`
Contains the `BDHModelHandler` class. This utility script is responsible for:
- Safely loading PyTorch model weights (`.pt` checkpoints) into memory as singletons.
- Initializing the OpenAI `tiktoken` BPE tokenizer.
- Executing the complex tensor mathematics required for predictions, phase-rotated topological embeddings, and intermediate layer activation captures.

---

## 📡 API Endpoints

### `POST /predict`
Executes language modeling inference to predict the subsequent tokens.
- **Request Body:** `{"text": "string", "temperature": 0.8, "top_k": 20}`
- **Response:** Array of precise token ID probabilities ranked by confidence.

### `POST /embeddings`
Extracts internal token embeddings and normalizes them for frontend rendering.
- **Request Body:** `{"text": "string"}`
- **Response:** Raw decoded tokens associated with mapped 32D normalized coordinate spaces.

### `POST /neurons`
Intercepts the forward pass to capture raw neuron activations at a specified hidden layer.
- **Request Body:** `{"text": "string", "layer": int (1-6)}`
- **Response:** Activation mappings strictly for the final sequence token, structured by attention head and internal dimension matrices. This feeds the 3D Barabási-Albert visualizer directly.

---

## ⚙️ Model Loading Behavior

On startup, `utils.py` will attempt to locate the model weights via relative pathing targeting `../Models/BDH_model/checkpoints/ckpt_5000.pt`. 

If a checkpoint cannot be located locally, the handler outputs a console warning and employs a random-initialization fallback to ensure the API does not hard-crash during frontend developmental work. Ensure your checkpoints are generated and correctly placed for accurate scientific visualization.
