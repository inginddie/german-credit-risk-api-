import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
from fastapi import HTTPException
from jira_models import JiraIssueInput, JiraTimePrediction

MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
MODEL_NAME = "JiraTimePrediction"
MODEL_ALIAS = "production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Mapeo de equipos y tipos
TEAM_MAPPING = {'ADP': 0, 'EFI': 1, 'TRX': 2}
TIPO_MAPPING = {
    'Historia': 0,
    'Historia No Funcional ( Habilitadora)': 1,
    'Incidente Produccion': 2,
    'Spike': 3,
    'Tarea': 4,
    'Test Case': 5,
    'Test Execution': 6,
    'Test Plan': 7,
    'Test Set': 8,
    'Xray Test': 9
}

try:
    logging.info(f"Cargando modelo {MODEL_NAME}@{MODEL_ALIAS}...")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    logging.info("Modelo Jira cargado correctamente")
except Exception as e:
    logging.error(f"Error cargando el modelo Jira: {e}")
    model = None

def predecir_tiempo_jira(issue: JiraIssueInput) -> JiraTimePrediction:
    if model is None:
        logging.error("Modelo no cargado")
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Preparar features
        team_encoded = TEAM_MAPPING.get(issue.team, 0)
        tipo_encoded = TIPO_MAPPING.get(issue.tipo_de_issue, 0)
        
        # Crear DataFrame con las features en el orden correcto
        data = pd.DataFrame([{
            'team_encoded': team_encoded,
            'tipo_encoded': tipo_encoded,
            'story_points': issue.story_points,
            'sprint_numbers': issue.sprint_numbers
        }])
        
        # Predecir
        tiempo_horas = float(model.predict(data)[0])
        # Asegurar que el tiempo no sea negativo (mínimo 1 hora)
        tiempo_horas = max(tiempo_horas, 1.0)
        tiempo_dias = tiempo_horas / 24
        
        # Determinar nivel de confianza y recomendación
        if tiempo_dias <= 7:
            nivel = "Alta"
            recomendacion = "Issue simple, desarrollo rápido esperado"
        elif tiempo_dias <= 20:
            nivel = "Media"
            recomendacion = "Issue estándar, seguimiento normal"
        elif tiempo_dias <= 40:
            nivel = "Media-Baja"
            recomendacion = "Issue compleja, requiere planificación detallada"
        else:
            nivel = "Baja"
            recomendacion = "Issue muy compleja, considerar dividir en subtasks"
        
        return JiraTimePrediction(
            equipo=issue.team,
            tipo_issue=issue.tipo_de_issue,
            story_points=issue.story_points,
            sprints=issue.sprint_numbers,
            tiempo_estimado_horas=round(tiempo_horas, 2),
            tiempo_estimado_dias=round(tiempo_dias, 2),
            nivel_confianza=nivel,
            recomendacion=recomendacion
        )
    
    except Exception as e:
        logging.error(f"Error en predicción Jira: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción")
