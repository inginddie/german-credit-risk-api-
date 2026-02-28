import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar MLflow
MLFLOW_TRACKING_URI = "http://44.211.88.225:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Jira-Model-Comparison")

print("=" * 80)
print("EXPERIMENTACIÓN DE MODELOS PARA PREDICCIÓN DE TIEMPO DE JIRA")
print("=" * 80)

# Cargar datos
print("\n1. Cargando y preparando datos...")
df = pd.read_csv('c:/Users/ingin/OneDrive/Documentos/Desarrollo/MlOps/jira_dataset.csv')
print(f"   Total registros: {len(df):,}")

# Preparar features
features = ['team', 'tipo_de_issue', 'story_points', 'sprint_numbers']
target = 'tiempo_desarrollo_horas'

df_model = df[features + [target]].copy()
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

X = df_model[['team_encoded', 'tipo_encoded', 'story_points', 'sprint_numbers']]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Definir modelos a experimentar
modelos = {
    'RandomForest_100': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'RandomForest_200': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'RandomForest_50': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
    
    'GradientBoosting_100': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'GradientBoosting_200': GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42),
    'GradientBoosting_50': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.2, random_state=42),
    
    'XGBoost_100': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
    'XGBoost_200': XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1),
    
    'AdaBoost_50': AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
    'AdaBoost_100': AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
    
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

resultados = []

print(f"\n2. Experimentando con {len(modelos)} configuraciones de modelos...")
print("=" * 80)

for i, (nombre, modelo) in enumerate(modelos.items(), 1):
    print(f"\n[{i}/{len(modelos)}] Entrenando {nombre}...")
    
    with mlflow.start_run(run_name=nombre):
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Cross-validation (3-fold para rapidez)
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=3, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        
        print(f"   MAE Train: {mae_train:.2f} h | Test: {mae_test:.2f} h | CV: {cv_mae:.2f} h")
        print(f"   RMSE Test: {rmse_test:.2f} h | R² Test: {r2_test:.4f}")
        
        # Log métricas
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_metric("cv_mae", cv_mae)
        
        # Calcular overfitting score
        overfitting = abs(mae_train - mae_test) / mae_test * 100
        mlflow.log_metric("overfitting_pct", overfitting)
        
        # Log parámetros
        mlflow.log_params(modelo.get_params())
        
        # Log modelo
        mlflow.sklearn.log_model(modelo, "model")
        
        # Guardar resultados
        resultados.append({
            'modelo': nombre,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'cv_mae': cv_mae,
            'overfitting_pct': overfitting
        })

print("\n" + "=" * 80)
print("RESULTADOS COMPARATIVOS")
print("=" * 80)

# Ordenar por MAE test
df_resultados = pd.DataFrame(resultados).sort_values('mae_test')

print("\n📊 RANKING POR MAE (Mean Absolute Error) - Menor es mejor:")
print("-" * 80)
for idx, row in df_resultados.head(10).iterrows():
    print(f"{row['modelo']:30s} | MAE: {row['mae_test']:7.2f} h ({row['mae_test']/24:5.1f} días) | "
          f"R²: {row['r2_test']:6.4f} | CV: {row['cv_mae']:7.2f} h | "
          f"Overfit: {row['overfitting_pct']:5.1f}%")

print("\n" + "=" * 80)
print(f"✅ MEJOR MODELO: {df_resultados.iloc[0]['modelo']}")
print(f"   MAE Test: {df_resultados.iloc[0]['mae_test']:.2f} horas ({df_resultados.iloc[0]['mae_test']/24:.1f} días)")
print(f"   RMSE Test: {df_resultados.iloc[0]['rmse_test']:.2f} horas")
print(f"   R² Test: {df_resultados.iloc[0]['r2_test']:.4f}")
print(f"   CV MAE: {df_resultados.iloc[0]['cv_mae']:.2f} horas")
print("=" * 80)

print(f"\n📊 Revisa los experimentos en MLflow UI:")
print(f"   {MLFLOW_TRACKING_URI}/#/experiments/")
print("\n⚠️  Siguiente paso: Ir a MLflow UI y asignar alias 'production' al mejor modelo")
