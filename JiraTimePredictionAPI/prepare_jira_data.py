import json
import pandas as pd
import numpy as np
from datetime import datetime

# Cargar datos
print("Cargando datos...")
data = json.load(open('c:/Users/ingin/OneDrive/Documentos/Desarrollo/jira_metrics_extractor/Archivos/combined_issues_20260127_213319.json', encoding='utf-8'))
df = pd.DataFrame(data)

print(f"Total registros: {len(df):,}\n")

# Convertir fechas
fecha_cols = ['created', 'fecha_desarrollo', 'fecha_qa', 'fecha_validacion', 'fecha_done', 'fecha_release', 'fecha_produccion']
for col in fecha_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calcular tiempos de desarrollo
df['tiempo_desarrollo_horas'] = (df['fecha_done'] - df['created']).dt.total_seconds() / 3600

# Filtrar registros con datos necesarios
df_modelo = df[
    (df['tiempo_desarrollo_horas'].notna()) & 
    (df['tiempo_desarrollo_horas'] > 0) &
    (df['team'].notna()) &
    (df['tipo_de_issue'].notna())
].copy()

print(f"Registros con tiempo de desarrollo: {len(df_modelo):,}")
print(f"\nEstadísticas de tiempo de desarrollo (horas):")
print(df_modelo['tiempo_desarrollo_horas'].describe())

print(f"\n\nDistribución por equipo:")
print(df_modelo['team'].value_counts())

print(f"\n\nDistribución por tipo de issue:")
print(df_modelo['tipo_de_issue'].value_counts())

print(f"\n\nTiempo promedio por equipo:")
print(df_modelo.groupby('team')['tiempo_desarrollo_horas'].agg(['mean', 'median', 'count']).round(2))

print(f"\n\nTiempo promedio por tipo de issue:")
print(df_modelo.groupby('tipo_de_issue')['tiempo_desarrollo_horas'].agg(['mean', 'median', 'count']).round(2))

# Guardar dataset procesado
df_modelo.to_csv('c:/Users/ingin/OneDrive/Documentos/Desarrollo/MlOps/jira_dataset.csv', index=False)
print(f"\n✅ Dataset guardado en: jira_dataset.csv")
