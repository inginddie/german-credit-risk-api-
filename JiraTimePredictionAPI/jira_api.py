from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jira_models import JiraIssueInput, JiraTimePrediction, BatchJiraPrediction
from jira_predict_logic import predecir_tiempo_jira, model, MODEL_NAME, MODEL_ALIAS
import uvicorn

app = FastAPI(
    title="Jira Time Prediction API",
    description="API de predicción de tiempo de desarrollo para issues de Jira — MLOps Project",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "api": "Jira Time Prediction API",
        "version": "1.0.0",
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "status": "running",
        "description": "Predicción de tiempo de desarrollo por equipo y tipo de issue"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }

@app.post("/predict/time", response_model=JiraTimePrediction)
def predict_time(issue: JiraIssueInput):
    """
    Predice el tiempo de desarrollo estimado para un issue de Jira.
    
    - **team**: Equipo asignado (ADP, TRX, EFI)
    - **tipo_de_issue**: Tipo de issue
    - **story_points**: Puntos de historia estimados (0-13)
    - **sprint_numbers**: Número de sprints (1-10)
    """
    return predecir_tiempo_jira(issue)

@app.post("/predict/time/batch", response_model=BatchJiraPrediction)
def predict_time_batch(issues: list[JiraIssueInput]):
    """
    Predice el tiempo de desarrollo para múltiples issues de Jira.
    """
    try:
        resultados = [predecir_tiempo_jira(issue) for issue in issues]
        return BatchJiraPrediction(total=len(resultados), predicciones=resultados)
    except HTTPException as e:
        raise e
    except Exception as e:
        import logging
        logging.error(f"Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción batch")

@app.get("/info/teams")
def get_teams():
    """Retorna la lista de equipos disponibles."""
    return {"teams": ["ADP", "TRX", "EFI"]}

@app.get("/info/issue-types")
def get_issue_types():
    """Retorna la lista de tipos de issues disponibles."""
    return {
        "issue_types": [
            "Historia",
            "Historia No Funcional ( Habilitadora)",
            "Incidente Produccion",
            "Tarea",
            "Spike",
            "Xray Test",
            "Test Execution",
            "Test Case",
            "Test Set",
            "Test Plan"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("jira_api:app", host="0.0.0.0", port=8001, reload=False)
