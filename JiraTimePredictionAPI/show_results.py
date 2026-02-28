import pandas as pd

resultados = [
    {'modelo': 'XGBoost_100', 'mae_test': 441.64, 'r2_test': 0.4024, 'cv_mae': 454.04},
    {'modelo': 'RandomForest_50', 'mae_test': 443.05, 'r2_test': 0.4137, 'cv_mae': 447.17},
    {'modelo': 'GradientBoosting_100', 'mae_test': 443.22, 'r2_test': 0.4072, 'cv_mae': 450.70},
    {'modelo': 'XGBoost_200', 'mae_test': 444.60, 'r2_test': 0.3943, 'cv_mae': 457.45},
    {'modelo': 'RandomForest_100', 'mae_test': 445.56, 'r2_test': 0.4036, 'cv_mae': 451.21},
    {'modelo': 'RandomForest_200', 'mae_test': 446.45, 'r2_test': 0.4009, 'cv_mae': 451.51},
    {'modelo': 'GradientBousting_200', 'mae_test': 447.46, 'r2_test': 0.3905, 'cv_mae': 457.45},
    {'modelo': 'GradientBoosting_50', 'mae_test': 450.48, 'r2_test': 0.3862, 'cv_mae': 441.54},
    {'modelo': 'DecisionTree', 'mae_test': 452.28, 'r2_test': 0.3801, 'cv_mae': 458.87},
]

df = pd.DataFrame(resultados).sort_values('mae_test')
print('=' * 85)
print('RANKING DE MODELOS EXPERIMENTADOS')
print('=' * 85)
print()
for idx, row in df.iterrows():
    dias = row['mae_test']/24
    print(f"{idx+1:2d}. {row['modelo']:25s} | MAE: {row['mae_test']:7.2f} h ({dias:5.1f} d) | R2: {row['r2_test']:.4f} | CV: {row['cv_mae']:.2f} h")
print()
print('=' * 85)
print(f"🏆 MEJOR MODELO: {df.iloc[0]['modelo']}")
print(f"   MAE Test: {df.iloc[0]['mae_test']:.2f} horas = {df.iloc[0]['mae_test']/24:.1f} días")
print(f"   R² Score: {df.iloc[0]['r2_test']:.4f}")
print(f"   CV MAE: {df.iloc[0]['cv_mae']:.2f} horas")
print('=' * 85)
print()
print("📊 Todos los modelos están registrados en MLflow UI:")
print("   http://44.211.88.225:5000/#/experiments/4")
print()
print("📌 Próximo paso: Asignar alias 'production' al mejor modelo en MLflow")
