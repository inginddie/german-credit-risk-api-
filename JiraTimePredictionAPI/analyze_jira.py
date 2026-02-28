import json
import pandas as pd

# Cargar datos
data = json.load(open('c:/Users/ingin/OneDrive/Documentos/Desarrollo/jira_metrics_extractor/Archivos/combined_issues_20260127_213319.json', encoding='utf-8'))
df = pd.DataFrame(data)

print('=== ANÁLISIS COMPLETO DE DATOS DE JIRA ===\n')
print(f'Total registros: {len(df):,}')
print(f'Total columnas: {len(df.columns)}\n')

print('Top 10 equipos:')
print(df['team'].value_counts().head(10))

print('\n\nTop 10 asignados:')
print(df['assignee'].value_counts().head(10))

print('\n\nRango de fechas de creación:')
df['created'] = pd.to_datetime(df['created'])
print(f'Desde: {df["created"].min()}')
print(f'Hasta: {df["created"].max()}')

print('\n\nPosibles casos de uso para ML:')
print('1. Predicción de story points basado en summary y tipo de issue')
print('2. Clasificación de tipo de issue basado en el resumen')
print('3. Predicción de tiempo de desarrollo basado en características')
print('4. Predicción de asignación de equipo/persona basado en issue')
