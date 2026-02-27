
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- models.py ---
# Modelos de entrada y salida para la API
from pydantic import BaseModel, Field

class ClienteInput(BaseModel):
   Age:              int   = Field(..., ge=18, le=100, example=35)
   Sex:              int   = Field(..., ge=0,  le=1,   example=1)
   Job:              int   = Field(..., ge=0,  le=3,   example=2)
   Housing:          int   = Field(..., ge=0,  le=2,   example=1)
   Saving_accounts:  int   = Field(..., ge=0,  le=4,   example=1)
   Checking_account: int   = Field(..., ge=0,  le=3,   example=1)
   Credit_amount:    float = Field(..., gt=0,          example=1500)
   Duration:         int   = Field(..., gt=0,          example=12)
   Purpose:          int   = Field(..., ge=0,  le=7,   example=4)

class PrediccionOutput(BaseModel):
   risk:             str
   probability_good: float
   probability_bad:  float
   recommendation:   str

class BatchPrediccionOutput(BaseModel):
   total: int
   predicciones: list[PrediccionOutput]

# --- predict_logic.py ---
# Lógica de predicción separada
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
from fastapi import HTTPException
from models import ClienteInput, PrediccionOutput

MLFLOW_TRACKING_URI = "http://32.192.36.226:5000"
MODEL_NAME          = "GermanCreditRisk-XGBoost"
MODEL_ALIAS         = "production"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
try:
   logging.info(f"Cargando modelo {MODEL_NAME}@{MODEL_ALIAS}...")
   model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
   logging.info("Modelo cargado correctamente")
except Exception as e:
   logging.error(f"Error cargando el modelo: {e}")
   model = None

def realizar_prediccion(cliente: ClienteInput) -> PrediccionOutput:
   if model is None:
      logging.error("Modelo no cargado")
      raise HTTPException(status_code=503, detail="Modelo no disponible")
   try:
      data  = pd.DataFrame([cliente.model_dump()])
      pred  = model.predict(data)[0]
      proba = model.predict_proba(data)[0]
      risk      = "good" if pred == 1 else "bad"
      prob_good = float(proba[1])
      prob_bad  = float(proba[0])
      if prob_good >= 0.75:
         recommendation = "Aprobar credito — bajo riesgo"
      elif prob_good >= 0.50:
         recommendation = "Revisar manualmente — riesgo moderado"
      else:
         recommendation = "Rechazar credito — alto riesgo"
      return PrediccionOutput(
         risk=risk,
         probability_good=round(prob_good, 4),
         probability_bad=round(prob_bad, 4),
         recommendation=recommendation
      )
   except Exception as e:
      logging.error(f"Error en predicción: {e}")
      raise HTTPException(status_code=500, detail="Error interno en la predicción")

# --- main.py (endpoints) ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import ClienteInput, PrediccionOutput, BatchPrediccionOutput
from predict_logic import realizar_prediccion, model, MODEL_NAME, MODEL_ALIAS
import uvicorn

app = FastAPI(
   title="German Credit Risk API",
   description="API de predicción de riesgo crediticio — Maestría IA USA",
   version="1.0.0"
)
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
   return realizar_prediccion(cliente)

@app.post("/predict/batch", response_model=BatchPrediccionOutput)
def predict_batch(clientes: list[ClienteInput]):
   from fastapi import HTTPException
   try:
      resultados = [realizar_prediccion(c) for c in clientes]
      return BatchPrediccionOutput(total=len(resultados), predicciones=resultados)
   except HTTPException as e:
      raise e
   except Exception as e:
      import logging
      logging.error(f"Error en predicción batch: {e}")
      raise HTTPException(status_code=500, detail="Error interno en la predicción batch")

@app.get("/")
def root():
   return {
      "api":    "German Credit Risk API",
      "version": "1.0.0",
      "model":  f"{MODEL_NAME}@{MODEL_ALIAS}",
      "status": "running"
   }

@app.get("/health")
def health():
   return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)