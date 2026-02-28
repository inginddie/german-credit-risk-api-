# Jira Time Prediction API

API REST para predicción de tiempo de desarrollo de issues de Jira utilizando Machine Learning, con experimentación de modelos en MLflow.

## 📋 Descripción

Este proyecto implementa una API de producción para predecir el tiempo de desarrollo estimado de issues de Jira basándose en características como equipo, tipo de issue, story points y número de sprints. Utiliza un modelo XGBoost entrenado y gestionado con MLflow.

## 📊 Datos

- **Dataset**: 10,024 issues de Jira de 3 equipos (ADP, TRX, EFI)
- **Período**: Enero 2024 - Marzo 2025
- **Features**: team, tipo_de_issue, story_points, sprint_numbers
- **Target**: tiempo_desarrollo_horas

## 🏗️ Arquitectura

- **jira_models.py**: Esquemas Pydantic para entrada y salida
- **jira_predict_logic.py**: Lógica de carga del modelo y predicción
- **jira_api.py**: Endpoints REST con FastAPI (puerto 8001)
- **prepare_jira_data.py**: Preparación y análisis exploratorio de datos
- **experiment_jira_models.py**: Experimentación con 14 configuraciones de modelos
- **train_jira_model.py**: Entrenamiento del modelo final
- **register_best_model.py**: Registro automático del mejor modelo en MLflow

## 🤖 Modelo en Producción

**Algoritmo**: XGBoost_100  
**MAE**: 441.64 horas (~18.4 días)  
**R² Score**: 0.4024  
**CV MAE**: 454.04 horas

Seleccionado tras experimentar con 14 configuraciones incluyendo:
- RandomForest (3 variantes)
- GradientBoosting (3 variantes)
- XGBoost (2 variantes)
- AdaBoost (2 variantes)
- DecisionTree, Ridge, Lasso, ElasticNet

## 📦 Endpoints

### `GET /`
Información de la API

### `GET /health`
Estado de salud y modelo cargado

### `POST /predict/time`
Predicción individual de tiempo de desarrollo

**Entrada:**
```json
{
  "team": "ADP",
  "tipo_de_issue": "Historia",
  "story_points": 5.0,
  "sprint_numbers": 1
}
```

**Salida:**
```json
{
  "equipo": "ADP",
  "tipo_issue": "Historia",
  "story_points": 5.0,
  "sprints": 1,
  "tiempo_estimado_horas": 872.52,
  "tiempo_estimado_dias": 36.36,
  "nivel_confianza": "Media-Baja",
  "recomendacion": "Issue compleja, requiere planificación detallada"
}
```

### `POST /predict/time/batch`
Predicción por lote para múltiples issues

### `GET /info/teams`
Lista de equipos disponibles

### `GET /info/issue-types`
Lista de tipos de issues disponibles

## 🔬 Experimentación

Para experimentar con diferentes modelos:

```bash
python experiment_jira_models.py
```

Esto entrenará 14 modelos diferentes y registrará métricas en MLflow:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross-Validation MAE
- Porcentaje de overfitting

## ▶️ Ejecución

```bash
python jira_api.py
```

La API estará disponible en: `http://localhost:8001`

Documentación interactiva: `http://localhost:8001/docs`

## 📊 Workflow Completo

1. **Preparar datos**: `python prepare_jira_data.py`
2. **Experimentar modelos**: `python experiment_jira_models.py`
3. **Ver resultados**: Revisar MLflow UI en http://44.211.88.225:5000/#/experiments/4
4. **Registrar mejor modelo**: `python register_best_model.py`
5. **Iniciar API**: `python jira_api.py`

## 🔧 Configuración

```python
MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
MODEL_NAME = "JiraTimePrediction"
MODEL_ALIAS = "production"
```

## 🎯 Niveles de Confianza

- **Alta** (≤ 7 días): Issue simple, desarrollo rápido esperado
- **Media** (8-20 días): Issue estándar, seguimiento normal
- **Media-Baja** (21-40 días): Issue compleja, requiere planificación
- **Baja** (> 40 días): Issue muy compleja, considerar dividir

## 🏆 Equipos Soportados

- ADP
- TRX
- EFI

## 📝 Tipos de Issues Soportados

- Historia
- Historia No Funcional (Habilitadora)
- Incidente Producción
- Tarea
- Spike
- Xray Test
- Test Execution
- Test Case
- Test Set
- Test Plan
