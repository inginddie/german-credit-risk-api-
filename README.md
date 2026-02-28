# MLOps - APIs de Machine Learning

Repositorio de proyectos de MLOps con APIs de predicción usando FastAPI y MLflow.

## 📁 Estructura del Proyecto

```
MlOps/
├── GermanCreditRiskAPI/       # API de predicción de riesgo crediticio
├── JiraTimePredictionAPI/     # API de predicción de tiempo de desarrollo
├── app.py                     # Archivo original (legacy)
├── mlops.ipynb               # Notebooks de experimentación
└── README.md                  # Este archivo
```

## 🚀 Proyectos

### 1. German Credit Risk API

API REST para predicción de riesgo crediticio de clientes.

**Características:**
- Modelo: XGBoost
- Dataset: German Credit Risk
- Puerto: 8000
- Endpoints: `/predict`, `/predict/batch`

**Documentación completa**: [GermanCreditRiskAPI/README.md](GermanCreditRiskAPI/README.md)

**Ejecutar:**
```bash
cd GermanCreditRiskAPI
python main.py
```

### 2. Jira Time Prediction API

API REST para predicción de tiempo de desarrollo de issues de Jira.

**Características:**
- Modelo: XGBoost (seleccionado tras experimentar con 14 modelos)
- Dataset: 10,024 issues de Jira (ADP, TRX, EFI)
- Puerto: 8001
- Endpoints: `/predict/time`, `/predict/time/batch`
- MAE: 441.64 horas (~18.4 días)

**Documentación completa**: [JiraTimePredictionAPI/README.md](JiraTimePredictionAPI/README.md)

**Ejecutar:**
```bash
cd JiraTimePredictionAPI
python jira_api.py
```

## 🛠️ Instalación General

1. Crear y activar un entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🔧 Configuración MLflow

Ambos proyectos usan MLflow para gestión de modelos:

**Servidor MLflow**: http://44.211.88.225:5000

Cada proyecto tiene su propio modelo registrado:
- **GermanCreditRisk-XGBoost** (German Credit Risk API)
- **JiraTimePrediction** (Jira Time Prediction API)

## 🏆 Buenas Prácticas Implementadas

- ✅ Separación de proyectos por dominio
- ✅ Código modularizado (modelos, lógica, endpoints)
- ✅ Validación de datos con Pydantic
- ✅ Logging estructurado
- ✅ Manejo robusto de errores
- ✅ Documentación automática con OpenAPI/Swagger
- ✅ Experimentación sistemática de modelos en MLflow
- ✅ Pruebas automáticas con pytest
- ✅ Type hints en Python
- ✅ Gestión de modelos con MLflow Model Registry

## 📊 MLflow UI

Accede a la interfaz de MLflow para:
- Ver experimentos y métricas
- Comparar modelos
- Gestionar versiones de modelos
- Asignar alias (staging, production)

**URL**: http://44.211.88.225:5000

## 📝 Workflow MLOps

1. **Preparación de datos** - Análisis exploratorio y limpieza
2. **Experimentación** - Probar múltiples modelos y configuraciones
3. **Registro en MLflow** - Tracking de métricas y parámetros
4. **Selección de modelo** - Comparar y elegir el mejor
5. **Promoción a producción** - Asignar alias 'production'
6. **Deployment** - Servir modelo via API REST
7. **Monitoreo** - Health checks y logging

## 👥 Autor

Desarrollado como parte de un proyecto de MLOps - Maestría en IA USA

**GitHub**: https://github.com/inginddie/german-credit-risk-api-
