from pydantic import BaseModel, Field
from typing import Literal

class JiraIssueInput(BaseModel):
    team: Literal["ADP", "TRX", "EFI"] = Field(..., example="ADP", description="Equipo asignado")
    tipo_de_issue: Literal[
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
    ] = Field(..., example="Historia", description="Tipo de issue")
    story_points: float = Field(5.0, ge=0, le=13, example=5.0, description="Story points estimados")
    sprint_numbers: int = Field(1, ge=1, le=10, example=1, description="Número de sprints")

class JiraTimePrediction(BaseModel):
    equipo: str
    tipo_issue: str
    story_points: float
    sprints: int
    tiempo_estimado_horas: float
    tiempo_estimado_dias: float
    nivel_confianza: str
    recomendacion: str

class BatchJiraPrediction(BaseModel):
    total: int
    predicciones: list[JiraTimePrediction]
