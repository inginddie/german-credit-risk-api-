from fastapi import FastAPI, HTTPException
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
