# German Credit Risk API

API REST para predicción de riesgo crediticio utilizando Machine Learning, desarrollada con FastAPI y MLflow.

## 📋 Descripción

Este proyecto implementa una API de producción para evaluar el riesgo crediticio de clientes basándose en el dataset German Credit Risk. Utiliza un modelo XGBoost entrenado y gestionado con MLflow, desplegado a través de una API RESTful construida con FastAPI.

## 🏗️ Arquitectura

El proyecto está modularizado en tres componentes principales:

- **models.py**: Define los esquemas de entrada y salida usando Pydantic
- **predict_logic.py**: Contiene la lógica de carga del modelo de MLflow y predicción
- **main.py**: Define los endpoints de la API con FastAPI
- **test_main.py**: Pruebas automáticas con pytest

## 🚀 Características

- ✅ Predicción individual de riesgo crediticio
- ✅ Predicción por lote (batch)
- ✅ Validación automática de datos de entrada
- ✅ Logging estructurado
- ✅ Manejo robusto de errores
- ✅ Documentación automática (Swagger/OpenAPI)
- ✅ Pruebas automáticas con pytest
- ✅ CORS habilitado

## 📦 Endpoints

### `GET /`
Información básica de la API y estado del modelo

### `GET /health`
Verificación del estado de salud de la API

### `POST /predict`
Predicción individual de riesgo crediticio

**Parámetros de entrada:**
- Age: Edad del cliente (18-100)
- Sex: Sexo (0: Femenino, 1: Masculino)
- Job: Tipo de trabajo (0-3)
- Housing: Situación de vivienda (0-2)
- Saving_accounts: Cuentas de ahorro (0-4)
- Checking_account: Cuenta corriente (0-3)
- Credit_amount: Monto del crédito (>0)
- Duration: Duración en meses (>0)
- Purpose: Propósito del crédito (0-7)

**Respuesta:**
```json
{
  "risk": "good",
  "probability_good": 0.8523,
  "probability_bad": 0.1477,
  "recommendation": "Aprobar credito — bajo riesgo"
}
```

### `POST /predict/batch`
Predicción por lote para múltiples clientes

## ▶️ Ejecución

```bash
python main.py
```

La API estará disponible en: `http://localhost:8000`

Documentación interactiva: `http://localhost:8000/docs`

## 🧪 Pruebas

Ejecutar las pruebas automáticas:

```bash
pytest test_main.py -v
```

## 🔧 Configuración

El modelo se carga desde un servidor MLflow configurado en `predict_logic.py`:

```python
MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
MODEL_NAME = "GermanCreditRisk-XGBoost"
MODEL_ALIAS = "production"
```

## 📊 Lógica de Recomendación

- **Probabilidad ≥ 75%**: Aprobar crédito — bajo riesgo
- **Probabilidad ≥ 50%**: Revisar manualmente — riesgo moderado
- **Probabilidad < 50%**: Rechazar crédito — alto riesgo
