import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar MLflow
MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Jira-Tiempo-Prediccion")

print("=" * 60)
print("MODELO DE PREDICCIÓN DE TIEMPO DE JIRA")
print("=" * 60)

# Cargar datos
print("\n1. Cargando datos...")
df = pd.read_csv('c:/Users/ingin/OneDrive/Documentos/Desarrollo/MlOps/jira_dataset.csv')
print(f"   Total registros: {len(df):,}")

# Seleccionar features relevantes
features = ['team', 'tipo_de_issue', 'story_points', 'sprint_numbers']
target = 'tiempo_desarrollo_horas'

# Preparar datos
print("\n2. Preparando features...")
df_model = df[features + [target]].copy()

# Rellenar story_points y sprint_numbers con mediana
df_model['story_points'] = df_model['story_points'].fillna(df_model['story_points'].median())
df_model['sprint_numbers'] = df_model['sprint_numbers'].fillna(1)

# Eliminar outliers extremos (> percentil 95)
percentil_95 = df_model[target].quantile(0.95)
df_model = df_model[df_model[target] <= percentil_95]
print(f"   Registros después de limpiar outliers: {len(df_model):,}")

# Codificar variables categóricas
le_team = LabelEncoder()
le_tipo = LabelEncoder()

df_model['team_encoded'] = le_team.fit_transform(df_model['team'])
df_model['tipo_encoded'] = le_tipo.fit_transform(df_model['tipo_de_issue'])

# Preparar X e y
X = df_model[['team_encoded', 'tipo_encoded', 'story_points', 'sprint_numbers']]
y = df_model[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Entrenar modelos
modelos = {
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

mejor_modelo = None
mejor_mae = float('inf')
mejor_nombre = None

print("\n3. Entrenando modelos...")
for nombre, modelo in modelos.items():
    print(f"\n   Entrenando {nombre}...")
    
    with mlflow.start_run(run_name=nombre):
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"   MAE Train: {mae_train:.2f} horas")
        print(f"   MAE Test: {mae_test:.2f} horas")
        print(f"   RMSE Test: {rmse_test:.2f} horas")
        print(f"   R² Test: {r2_test:.4f}")
        
        # Log métricas
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_test", r2_test)
        
        # Log parámetros
        mlflow.log_params(modelo.get_params())
        
        # Guardar encoders y features
        mlflow.log_dict({
            'team_classes': le_team.classes_.tolist(),
            'tipo_classes': le_tipo.classes_.tolist(),
            'features': features
        }, 'encoders.json')
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            modelo,
            "model",
            registered_model_name="JiraTimePrediction"
        )
        
        # Guardar mejor modelo
        if mae_test < mejor_mae:
            mejor_mae = mae_test
            mejor_modelo = modelo
            mejor_nombre = nombre

print("\n" + "=" * 60)
print(f"✅ MEJOR MODELO: {mejor_nombre}")
print(f"   MAE: {mejor_mae:.2f} horas (~{mejor_mae/24:.1f} días)")
print("=" * 60)

print(f"\n📊 Modelo registrado en MLflow: {MLFLOW_TRACKING_URI}")
print("   Nombre: JiraTimePrediction")
print("\n⚠️  Siguiente paso: Asignar alias 'production' en MLflow UI")
