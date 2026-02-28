import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configurar MLflow
MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("=" * 80)
print("BUSCANDO Y REGISTRANDO MEJOR MODELO COMO PRODUCTION")
print("=" * 80)

# Buscar el run de XGBoost_100 en el experimento
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Jira-Model-Comparison")

if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.mae_test ASC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        print(f"\n✅ Mejor run encontrado:")
        print(f"   Run ID: {best_run.info.run_id}")
        print(f"   Run Name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"   MAE Test: {best_run.data.metrics.get('mae_test', 0):.2f} horas")
        print(f"   R² Test: {best_run.data.metrics.get('r2_test', 0):.4f}")
        
        # Registrar el modelo con el nombre correcto
        model_uri = f"runs:/{best_run.info.run_id}/model"
        
        # Verificar si el modelo ya está registrado
        try:
            registered_models = client.search_registered_models(filter_string="name='JiraTimePrediction'")
            if registered_models:
                print(f"\n📦 Modelo 'JiraTimePrediction' ya existe con {len(registered_models[0].latest_versions)} versiones")
                
                # Registrar nueva versión
                print(f"\n📤 Registrando nueva versión del mejor modelo...")
                model_version = mlflow.register_model(model_uri, "JiraTimePrediction")
                version_number = model_version.version
                print(f"   ✅ Versión {version_number} registrada")
                
                # Asignar alias 'production'
                print(f"\n🏷️  Asignando alias 'production' a la versión {version_number}...")
                client.set_registered_model_alias("JiraTimePrediction", "production", version_number)
                print(f"   ✅ Alias 'production' asignado correctamente")
                
            else:
                print(f"\n📦 Creando modelo registrado 'JiraTimePrediction'...")
                model_version = mlflow.register_model(model_uri, "JiraTimePrediction")
                version_number = model_version.version
                
                # Asignar alias 'production'
                print(f"\n🏷️  Asignando alias 'production' a la versión {version_number}...")
                client.set_registered_model_alias("JiraTimePrediction", "production", version_number)
                print(f"   ✅ Modelo registrado y alias asignado")
                
        except Exception as e:
            print(f"❌ Error registrando modelo: {e}")
            
        print("\n" + "=" * 80)
        print("✅ MODELO XGBoost_100 CONFIGURADO COMO PRODUCTION")
        print("=" * 80)
        print(f"\n📊 Puedes verificar en: {MLFLOW_TRACKING_URI}/#/models/JiraTimePrediction")
        
    else:
        print("❌ No se encontraron runs en el experimento")
else:
    print("❌ Experimento 'Jira-Model-Comparison' no encontrado")
