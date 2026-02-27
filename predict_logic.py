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
